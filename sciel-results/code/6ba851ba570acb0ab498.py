import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    
    # 1. Initialize with 8 disjoint K5s (104 edges)
    # This is a strong C6-free base.
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append((5 * k + a, 5 * k + b))
    
    # Bitset representation for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    current_edges = set(edges)
    best_edges = set(edges)
    best_score = len(best_edges)
    
    # 2. Generate candidate edges (inter-clique edges)
    # We only need to consider edges between different K5s because
    # intra-clique edges are already present.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u // 5) != (v // 5):
                candidates.append((u, v))
    
    # 3. Define a fast C6 check
    # A C6 containing (u, v) exists iff there is a path of length 4 from u to v
    # in the graph G - {u, v} (but since (u,v) is not in G yet, just in G).
    # Path: u-w-x-y-z-v
    # We can check this by iterating w in N(u), then checking if there is a path
    # of length 3 from w to v in G - {u, v, w}.
    # Path w-x-y-z-v:
    # Iterate x in N(w), check if there is y in N(x) such that N(y) intersects N(v) \ {u, w, x, y}.
    
    def has_c6_fast(u, v):
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # w in N(u) \ {v}
        w_mask = mask_u & ~(1 << v)
        
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # x in N(w) \ {u, v, w}
            mask_w = adj_mask[w]
            x_mask = mask_w & ~(1 << u) & ~(1 << v) & ~(1 << w)
            
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # y in N(x) \ {u, v, w, x}
                mask_x = adj_mask[x]
                y_mask = mask_x & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # z in N(y) \ {u, v, w, x, y} AND z in N(v)
                    mask_y = adj_mask[y]
                    z_mask = mask_y & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    
                    if z_mask & mask_v:
                        return True
        return False

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        current_edges.add((u, v))

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        current_edges.remove((u, v))

    # 4. Heuristic: Score candidates by "risk"
    # Risk is the number of paths of length 4 between u and v in the current graph.
    # If risk is 0, adding the edge is safe.
    # If risk > 0, it creates a C6.
    # We want to add edges with risk 0.
    # To speed up, we can just check has_c6_fast.
    # But to prioritize, we can sort by degree or some other metric.
    # Let's just sort candidates by (u, v) to be deterministic, 
    # or maybe by degree sum to try "central" edges first?
    # Actually, random order is often better to avoid bias.
    # But deterministic is required for reproducibility in some contexts?
    # The task says "exact search" but the interface allows any method.
    # The current program uses random. Let's use a deterministic 
    # "greedy with backtracking" or just "greedy" approach.
    
    # Since we have a time limit, a simple greedy might get stuck.
    # Let's try a greedy approach that tries to add edges in a specific order.
    # If we can't add an edge, we skip it.
    # This is fast but might not be optimal.
    
    # Better: Use the time to do a randomized greedy search with 
    # multiple restarts or a "k-best" selection.
    
    # Let's try a strategy:
    # 1. Sort candidates by some heuristic (e.g., random, or degree).
    # 2. Iterate through them. If has_c6_fast is False, add it.
    # 3. If we run out of candidates, we are done.
    # 4. To improve, we can do multiple passes with different orderings.
    
    # Let's implement a "Greedy with Randomized Order" loop.
    # We will perform multiple "greedy sweeps".
    # In each sweep, we shuffle the remaining candidates (or all candidates)
    # and try to add them.
    
    # However, shuffling all candidates every time is O(E log E).
    # With ~760 candidates, it's cheap.
    
    # Strategy:
    # While time remains:
    #   Shuffle candidates
    #   For each candidate in shuffled list:
    #     If not in current_edges:
    #       If has_c6_fast(u, v) is False:
    #         Add edge
    #         Update best if improved
    #   If no edge was added in the entire sweep, break (local optimum)
    #   (Optional) Randomly remove a few edges to escape local optimum?
    
    # Let's implement this.
    
    random.seed(42) # For reproducibility if needed, but task allows random
    
    no_progress_streak = 0
    
    while time.time() < deadline:
        # Shuffle candidates to change the order of attempts
        random.shuffle(candidates)
        
        progress = 0
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
                
            # Check if adding (u, v) creates a C6
            if not has_c6_fast(u, v):
                add_edge(u, v)
                progress += 1
                if len(current_edges) > best_score:
                    best_score = len(current_edges)
                    best_edges = set(current_edges)
        
        if progress == 0:
            # We are stuck in a local optimum
            # Try to escape by removing a random edge
            if current_edges:
                # Pick a random edge to remove
                # To be smart, pick an edge that might be "blocking" many others?
                # For now, just random.
                e = random.choice(list(current_edges))
                remove_edge(e[0], e[1])
                no_progress_streak += 1
                
                # Limit restarts?
                if no_progress_streak > 100:
                    break
            else:
                break
        else:
            no_progress_streak = 0
            
    return [[a, b] for (a, b) in sorted(best_edges)]
