import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    n = 18
    full_mask = (1 << n) - 1
    
    # Helper: Check if bitmask represents an independent set in current graph adj
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    # Exact B&B for largest triangle-free induced subgraph
    def largest_tf(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic to get a good initial lower bound
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        
        greedy_S = 0
        for v in order:
            if is_independent(adj[v] & greedy_S, adj):
                greedy_S |= (1 << v)
        best[0] = bin(greedy_S).count('1')
        
        if best[0] == n:
            return n
            
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            
            # Prune 1
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Prune 2: Remove candidates that cannot be added
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb
            
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return
                
            if new_C != C_mask:
                C_mask = new_C
                if sz + bin(C_mask).count('1') <= best[0]:
                    return

            # Branching heuristic: pick vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v
            bnb(S_mask | v_bit, C_mask & ~v_bit)

        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # --- Structural Initialization: 3-partite Turan Graph ---
    # A 3-partite graph where each part is a clique is K4-free.
    # We start with parts of size 6.
    # This structure has specific triangle-free properties that might be optimized.
    
    def make_turan_adj(parts):
        # parts: list of lists of vertices
        adj = [0] * n
        for p in parts:
            for i in range(len(p)):
                for j in range(len(p)):
                    if i != j:
                        adj[p[i]] |= (1 << p[j])
        return adj

    # Initial parts
    parts = [
        list(range(0, 6)),
        list(range(6, 12)),
        list(range(12, 18))
    ]
    
    adj = make_turan_adj(parts)
    
    # Verify K4-free (should be, but good to be safe with the helper)
    # Actually, we know it is.
    
    current_score = largest_tf(adj, time_limit=deadline - 10.0)
    best_score = current_score
    best_adj = adj[:]
    best_parts = [p[:] for p in parts]

    random.seed(123)
    
    # Local Search: Swap vertices between parts
    # Swapping vertices between parts preserves the "3-partite clique" structure, 
    # thus maintaining K4-freeness.
    # We want to minimize the largest triangle-free induced subgraph.
    
    T = 5.0
    T_min = 0.01
    alpha = 0.99
    
    all_vertices = list(range(n))
    
    while time.time() < deadline - 10.0 and T > T_min:
        T *= alpha
        
        # Pick two distinct parts
        p1, p2 = random.sample([0, 1, 2], 2)
        
        # Pick a vertex from p1 and a vertex from p2
        if not parts[p1] or not parts[p2]:
            continue
            
        v1 = random.choice(parts[p1])
        v2 = random.choice(parts[p2])
        
        if v1 == v2:
            continue
        
        # Perform swap
        parts[p1].remove(v1)
        parts[p1].append(v2)
        parts[p2].remove(v2)
        parts[p2].append(v1)
        
        # Update adjacency
        # Removing edges from old neighbors, adding to new
        # Old neighbors of v1 in p1 (except v2 if it was there, but they are from diff parts so no)
        # v1 was in p1. It was connected to all in p1 except itself.
        # v2 was in p2. It was connected to all in p2 except itself.
        
        # Let's just rebuild adj for simplicity if n is small? 
        # n=18, rebuilding is fast. 
        # Or do incremental.
        
        # Incremental update:
        # v1 loses edges to (p1 \ {v1}) and gains edges to (p2 \ {v2})
        # v2 loses edges to (p2 \ {v2}) and gains edges to (p1 \ {v1})
        
        # Note: v1 and v2 are now in each other's old parts.
        # They do NOT connect to each other because they are in different parts now.
        # In the original state, they were in different parts, so they did NOT connect.
        # So the edge (v1, v2) status doesn't change? 
        # Wait.
        # Before: v1 in P1, v2 in P2. Edge(v1, v2) = False.
        # After: v1 in P2, v2 in P1. Edge(v1, v2) = False.
        # So edge (v1, v2) remains False.
        
        # For any other vertex u:
        # If u was in P1:
        #   Before: Edge(u, v1) = True (if u!=v1), Edge(u, v2) = False.
        #   After: Edge(u, v1) = False (v1 moved to P2), Edge(u, v2) = True (v2 moved to P1).
        # If u was in P2:
        #   Before: Edge(u, v1) = False, Edge(u, v2) = True (if u!=v2).
        #   After: Edge(u, v1) = True (v1 moved to P2), Edge(u, v2) = False.
        # If u was in P3:
        #   No change.
        
        # So we need to flip edges between v1 and (P1 \ {v1, v2}) and (P2 \ {v1, v2})?
        # Actually:
        # v1's neighbors:
        #   Old: P1 \ {v1}
        #   New: P2 \ {v2}
        # v2's neighbors:
        #   Old: P2 \ {v2}
        #   New: P1 \ {v1}
        
        # Let's just reconstruct adj from parts to avoid bugs. n=18 is tiny.
        adj_new = make_turan_adj(parts)
        
        new_score = largest_tf(adj_new, time_limit=deadline - 10.0)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_score = new_score
            adj = adj_new
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                best_parts = [p[:] for p in parts]
        else:
            # Reject: Swap back
            parts[p1].remove(v2)
            parts[p1].append(v1)
            parts[p2].remove(v1)
            parts[p2].append(v2)
            
    edges = get_edge_list(best_adj)
    return edges
