import time
import random
import numpy as np
import sys

def construct():
    """
    Constructs a C6-free graph on 40 vertices with many edges.
    Strategy: Simulated Annealing with edge swaps (1-out, 1-in).
    - Starts from 8 disjoint K5s (80 edges).
    - Uses NumPy bitsets for fast C6 feasibility checks.
    - Accepts worse moves probabilistically to escape local optima.
    """
    n = 40
    start_time = time.time()
    deadline = time.time() + 115  # Safety margin: 150s limit, stop at 115s to return safely
    
    # 1. Initialize graph: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = []
    edge_set = set()
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                edge_set.add((u, v))
                edge_set.add((v, u))
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper: Check if adding edge (u, v) creates a C6
    # We assume the graph is currently C6-free or we are checking a specific move.
    # For SA, we need to check if the *resulting* graph has a C6.
    # Efficient check: A C6 is created if there is a path of length 4 between u and v
    # in the graph *without* the edge (u,v).
    # Path: u - x - y - z - v
    # This requires:
    # 1. x in N(u)
    # 2. z in N(v)
    # 3. y in N(x) and y in N(z)
    # 4. x, y, z distinct and not u, v.
    
    def creates_c6(u, v):
        # N(u) and N(v) as lists of neighbors
        # Using numpy for intersection is fast for small n, but list comp might be faster for very sparse?
        # n=40, so sets/lists are small.
        nu = np.nonzero(adj[u])[0]
        nv = np.nonzero(adj[v])[0]
        
        # We need x in nu, z in nv, and common neighbor y of x and z.
        # Optimization: Iterate over the smaller of nu/nv? No, structure is fixed.
        # Iterate x in nu
        for x in nu:
            if x == v: continue
            # Iterate z in nv
            for z in nv:
                if z == u or z == x: continue
                # Check if x and z share a neighbor y
                # y must be in adj[x] and adj[z]
                # and y != u, v, x, z
                # Using numpy intersection
                common = np.intersect1d(np.nonzero(adj[x])[0], np.nonzero(adj[z])[0])
                for y in common:
                    if y == u or y == v or y == x or y == z:
                        continue
                    return True
        return False
    
    # Better C6 check using bitsets for speed
    def creates_c6_bitset(u, v):
        nu = adj[u]
        nv = adj[v]
        # Iterate x in neighbors of u
        # To speed up, we can precompute neighbor lists as integers (bitmasks)
        # Let's use Python integers as bitmasks for the 40 vertices.
        # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge.
        
        # Wait, I haven't built adj_mask. Let's build it once.
        pass

    # Let's rebuild using integer bitmasks for maximum speed in the SA loop.
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    def creates_c6_fast(u, v):
        # Check if path u-x-y-z-v exists
        # x in N(u), z in N(v), y in N(x) & N(z)
        # nu_mask = adj_mask[u]
        # nv_mask = adj_mask[v]
        
        # Iterate through bits of adj_mask[u]
        nu_mask = adj_mask[u]
        nv_mask = adj_mask[v]
        
        # We need to iterate x in nu, z in nv.
        # To make this fast, we can iterate over the smaller set?
        # Or just iterate all x in nu.
        # Since degree is ~5 initially, and grows to ~5-6, this is very fast.
        
        # Get list of neighbors
        nu = []
        m = nu_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nu.append(idx)
            m ^= lsb
            
        nv = []
        m = nv_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nv.append(idx)
            m ^= lsb
            
        for x in nu:
            if x == v: continue
            for z in nv:
                if z == u or z == x: continue
                # Check common neighbors of x and z
                common = adj_mask[x] & adj_mask[z]
                # Remove u, v, x, z from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                common &= ~(1 << x)
                common &= ~(1 << z)
                if common != 0:
                    return True
        return False

    # State
    current_edges = set(edges)
    current_count = len(current_edges)
    best_edges = list(current_edges)
    best_count = current_count
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    cooling = 0.995
    
    T = T_start
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        if steps % 1000 == 0 and T < T_end:
            break
            
        # Pick a random edge to remove (u, v)
        u, v = random.choice(list(current_edges))
        
        # Pick a random non-edge to add (a, b)
        # To avoid O(N^2) sampling, we can pick random a, b and check if edge exists.
        # Or maintain a list of non-edges. Maintaining list is O(1) update but O(N) init.
        # Let's just pick random pairs. With 780 edges, probability of hitting non-edge is high.
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if a == b:
            continue
        if a > b:
            a, b = b, a
            
        if (a, b) in current_edges:
            continue
            
        # Check if removing (u,v) and adding (a,b) results in a C6-free graph.
        # We need to check if the NEW graph has a C6.
        # The new graph is G - (u,v) + (a,b).
        # Any new C6 must use the edge (a,b).
        # So we check if there is a path of length 4 between a and b in G - (u,v).
        
        # Temporarily remove (u,v) from adj_mask
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
        # Check if adding (a,b) creates a C6 in the current state (which is G - (u,v))
        is_valid = not creates_c6_fast(a, b)
        
        # If valid, the move is accepted (improvement or neutral in terms of validity)
        # Wait, SA usually minimizes energy. Here we want to maximize edges.
        # If the move is valid (C6-free), we accept it.
        # If the move is invalid (creates C6), we might accept it with probability exp(-delta/T)?
        # No, the constraint is hard: "containing no six-cycle". We cannot return an invalid graph.
        # So we can only accept valid moves.
        # This makes it a Random Walk on the space of C6-free graphs.
        # To escape local optima, we can try "2-swap" moves or just rely on the random walk mixing well.
        # Actually, if we only accept valid moves, we are doing a random walk.
        # We should accept valid moves that increase edge count, and with some probability, accept valid moves that keep edge count same (neutral) to explore.
        # We should NOT accept moves that decrease edge count unless we are doing a specific annealing on "validity" which we can't because invalid is forbidden.
        
        # So:
        # 1. If is_valid:
        #    - The move keeps edge count same (1 out, 1 in).
        #    - We accept it always? Or with some prob?
        #    - If we accept always, we just do a random walk.
        #    - To find *more* edges, we need to find a sequence of neutral moves that leads to a state where an *additive* move is possible.
        #    - So we should also try *additive* moves (0 out, 1 in) if possible.
        
        # Let's combine:
        # 1. Try to ADD an edge (0 out, 1 in). If valid, accept.
        # 2. If no add possible, try SWAP (1 out, 1 in). If valid, accept with prob p (to explore).
        
        # Revised Loop:
        
        # 1. Try Add
        added = False
        # Pick random non-edge
        for _ in range(5): # Try a few random additions
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b: continue
            if a > b: a, b = b, a
            if (a, b) in current_edges: continue
            
            # Check if adding (a,b) creates C6
            if not creates_c6_fast(a, b):
                # Accept
                current_edges.add((a, b))
                current_count += 1
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                added = True
                if current_count > best_count:
                    best_count = current_count
                    best_edges = list(current_edges)
                break # Restart loop to try another add? Or just continue?
        
        if added:
            # If we added, we might have unlocked new swaps or adds.
            # Continue to next iteration
            T *= cooling
            continue
            
        # 2. If no add found, try Swap
        # Pick random edge (u,v) to remove
        u, v = random.choice(list(current_edges))
        
        # Pick random non-edge (a,b) to add
        # Ensure (a,b) != (u,v)
        while True:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b: continue
            if a > b: a, b = b, a
            if (a, b) in current_edges: continue
            if (a, b) == (u, v) or (a, b) == (v, u): continue
            break
        
        # Perform swap: Remove (u,v), Add (a,b)
        # Check validity of new graph
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
        if not creates_c6_fast(a, b):
            # Valid swap. Accept it.
            # This is a neutral move (edge count same).
            # We accept it to explore the space.
            current_edges.discard((u, v))
            current_edges.add((a, b))
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            # Update best? No, count is same.
            # But this new state might allow future adds.
            
        else:
            # Invalid swap. Revert.
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        T *= cooling

    # Return the best graph found
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
