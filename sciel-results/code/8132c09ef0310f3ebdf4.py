import time
import random
import sys
from collections import defaultdict

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Helper to check if a graph is C6-free
    # Using bitsets for speed: adj[i] is an integer where bit j is 1 if edge (i,j) exists
    # A C6 exists if there is a path of length 4 between any non-adjacent pair (u,v).
    # Path u-x-y-z-v.
    # We can compute the matrix of paths of length 2, 3, 4 using bitset operations.
    
    def is_c6_free(adj_bits):
        # adj_bits: list of n integers
        # Compute paths of length 2: P2[u] = union of N(v) for v in N(u)
        # But we need to be careful with simple paths.
        # For a fast check, we can just try to find a C6.
        # A C6 is u-v-w-x-y-z-u.
        # We can iterate over all pairs (u,v) that are NOT adjacent.
        # If there is a path of length 4 between u and v, we have a C6 (u...v + edge v-u? No, u,v not adjacent).
        # Wait. If u and v are not adjacent, and there is a path of length 4 between them, 
        # then adding the edge (u,v) would create a C6. 
        # But we are checking if the CURRENT graph contains a C6.
        # A C6 in the current graph means there exist 6 distinct vertices forming a cycle.
        # This is equivalent to: there exist u, v such that u, v are adjacent, 
        # and there is a path of length 4 between u and v that does not use the edge (u,v).
        # Or simply: there exist u, v (adjacent or not) such that there is a path of length 4 between them?
        # No. A C6 is a cycle. 
        # Let's use a simpler check for the search: 
        # When adding an edge (u,v) to a C6-free graph, we check if a path of length 4 exists between u and v.
        # For the initial random graph, we just verify it's C6-free once.
        
        # Efficient C6 check for a static graph:
        # For each pair (u,v) with u < v:
        # If edge (u,v) exists:
        #   Check if there is a path of length 4 between u and v avoiding the edge (u,v).
        #   This is hard to do quickly with simple bitsets without matrix mult.
        
        # Alternative: Just run a BFS from each node to depth 3?
        # Or, since we start with a random 4-regular graph, it is very likely C6-free.
        # We can use a randomized check or a simple backtracking check if needed.
        # But for the local search, we only need to check if ADDING an edge creates a C6.
        # So we need a fast `would_create_c6(u, v, adj_bits)`.
        
        # For the initial verification, we can just assume it's C6-free if it's random 4-reg?
        # No, we must be sure. 
        # Let's implement a fast C6 detection for the initial graph.
        # A graph has a C6 if there is a path of length 4 between two adjacent vertices?
        # No, that creates a C5 or C6? 
        # If u-v is an edge, and there is a path u-x-y-z-v of length 4, then u-x-y-z-v-u is a C6.
        # So: For every edge (u,v), check if there is a path of length 4 between u and v.
        
        # Path of length 4 between u and v:
        # u - x - y - z - v
        # x in N(u), z in N(v), y in N(x) intersect N(z).
        # x != z, x != v, z != u, y distinct from others.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1: # Edge exists
                    # Check for path of length 4
                    nu = adj_bits[u]
                    nv = adj_bits[v]
                    # Iterate x in N(u)
                    # x cannot be v
                    x_mask = nu & ~(1 << v)
                    while x_mask:
                        x_bit = x_mask & -x_mask
                        x = x_bit.bit_length() - 1
                        x_mask -= x_bit
                        
                        # Iterate z in N(v)
                        # z cannot be u
                        # z cannot be x
                        z_mask = nv & ~(1 << u) & ~(1 << x)
                        while z_mask:
                            z_bit = z_mask & -z_mask
                            z = z_bit.bit_length() - 1
                            z_mask -= z_bit
                            
                            # Check if N(x) and N(z) have a common neighbor y
                            # y cannot be u, v, x, z
                            common = adj_bits[x] & adj_bits[z]
                            # Exclude u, v, x, z
                            common &= ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << z)
                            if common:
                                return False
        return True

    def would_create_c6(u, v, adj_bits):
        # Check if adding (u,v) creates a C6.
        # This requires a path of length 4 between u and v in the current graph.
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        x_mask = nu
        while x_mask:
            x_bit = x_mask & -x_mask
            x = x_bit.bit_length() - 1
            x_mask -= x_bit
            
            # z in N(v), z != u, z != x
            z_mask = nv & ~(1 << u) & ~(1 << x)
            while z_mask:
                z_bit = z_mask & -z_mask
                z = z_bit.bit_length() - 1
                z_mask -= z_bit
                
                common = adj_bits[x] & adj_bits[z]
                common &= ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << z)
                if common:
                    return True
        return False

    def random_4_regular():
        # Generate a random 4-regular graph on 40 vertices
        # Using configuration model
        while True:
            # Create stubs
            stubs = []
            for i in range(n):
                for _ in range(4):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj_bits = [0] * n
            edges = set()
            valid = True
            
            # Pair up stubs
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
                edges.add((v, u))
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            if not valid:
                continue
                
            # Check degrees
            deg_ok = True
            for i in range(n):
                if bin(adj_bits[i]).count('1') != 4:
                    deg_ok = False
                    break
            if not deg_ok:
                continue
                
            # Check C6-free
            if is_c6_free(adj_bits):
                return adj_bits
                
    # Main search loop
    best_edges = []
    best_count = 0
    
    # Initial guess: 80 edges
    # We will try to improve from random 4-regular starts
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        adj_bits = random_4_regular()
        if adj_bits is None:
            continue
            
        current_count = n * 4 // 2 # 80
        
        # Local search: Edge swaps and additions
        # We maintain a list of non-edges
        present_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    present_set.add((i, j))
        
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in present_set:
                    non_edges.append((i, j))
        
        # Randomize non_edges
        random.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add an edge
            # We iterate through non_edges. If we can add one, we add it, remove from non_edges, and restart scan?
            # Or just do one pass.
            
            # To be efficient, we check candidates in random order.
            # If we add an edge, we update adj_bits and present_set.
            
            # We also try swaps? 
            # A swap: remove (a,b), add (c,d).
            # This maintains edge count. Useful if we are stuck at a local maximum for additions.
            # But first, let's see if we can just add edges.
            
            for idx in range(len(non_edges)):
                if time.time() > deadline - 5:
                    break
                    
                u, v = non_edges[idx]
                # Check if still non-edge (might have been added by a swap? No, we only add)
                # But if we added it, it's no longer in non_edges? 
                # We are iterating a static list. We should check if it's already present.
                if (u, v) in present_set or (v, u) in present_set:
                    continue
                    
                if not would_create_c6(u, v, adj_bits):
                    # Add edge
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    present_set.add((u, v))
                    present_set.add((v, u))
                    current_count += 1
                    improved = True
                    # Remove from non_edges? 
                    # We can't easily remove from list during iteration.
                    # But we can just let the check `if (u,v) in present_set` handle it in next passes.
                    # However, adding an edge might invalidate other non_edges (make them create C6).
                    # That's fine, they will just fail the check.
                    
                    if current_count > best_count:
                        best_count = current_count
                        # Extract edges for return
                        best_edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if (adj_bits[i] >> j) & 1:
                                    best_edges.append([i, j])
                        
                        if best_count >= 105:
                            return best_edges
                            
            # If no edges could be added in this pass, try some random swaps?
            # Or just break and restart?
            # Swaps are complex. Let's just break and restart with a new random graph.
            # But maybe we can try a few random swaps to escape local optimum?
            
            # Simple swap heuristic:
            # Pick random edge (a,b), remove it.
            # Pick random non-edge (c,d), try to add it.
            # If successful, keep. Else revert.
            
            if not improved:
                # Try 100 random swaps
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                        
                    # Pick random edge
                    edges_list = list(present_set)
                    if not edges_list: break
                    a, b = random.choice(edges_list)
                    
                    # Remove
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
                    present_set.discard((a, b))
                    present_set.discard((b, a))
                    current_count -= 1
                    
                    # Pick random non-edge
                    # Re-generate non_edges or pick from list?
                    # Just pick random pair
                    found_swap = False
                    for _ in range(10): # Try 10 random non-edges
                        c = random.randint(0, n-1)
                        d = random.randint(0, n-1)
                        if c == d: continue
                        if c > d: c, d = d, c
                        if (c, d) in present_set: continue
                        if would_create_c6(c, d, adj_bits):
                            continue
                        
                        # Add
                        adj_bits[c] |= (1 << d)
                        adj_bits[d] |= (1 << c)
                        present_set.add((c, d))
                        present_set.add((d, c))
                        current_count += 1
                        found_swap = True
                        break
                        
                    if not found_swap:
                        # Revert
                        adj_bits[a] |= (1 << b)
                        adj_bits[b] |= (1 << a)
                        present_set.add((a, b))
                        present_set.add((b, a))
                        current_count += 1

    return best_edges

# EVOLVE-BLOCK-END
