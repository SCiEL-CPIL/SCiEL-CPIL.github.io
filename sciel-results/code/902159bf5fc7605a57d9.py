import time
import numpy as np
import itertools
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 110
    n = 30
    n_edges = n * (n - 1) // 2
    
    def build_adjacency(colors):
        adj0 = np.zeros((n, n), dtype=bool)
        adj1 = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i, j] = adj0[j, i] = True
                else:
                    adj1[i, j] = adj1[j, i] = True
                idx += 1
        return adj0, adj1

    def count_k222(adj):
        count = 0
        # Precompute neighbor sets as lists of indices for speed?
        # Or use bitmasks.
        # n=30 fits in an int.
        neighbors = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    neighbors[i] |= (1 << j)
        
        # For each pair u, v
        for u in range(n):
            for v in range(u + 1, n):
                # S = N(u) & N(v)
                mask = neighbors[u] & neighbors[v]
                # Remove u and v from mask?
                # u and v are not in N(u) or N(v) typically (no self loops), 
                # but if adj is symmetric and no self loops, u is not in N(u).
                # However, u is in N(v) if adj[u,v] is True.
                # The set S must be disjoint from {u,v} to form distinct parts?
                # Yes, the 6 vertices must be distinct.
                mask &= ~(1 << u)
                mask &= ~(1 << v)
                
                # Count K2,2 in subgraph induced by mask
                # Iterate over pairs x, y in mask
                # We need to extract indices from mask.
                # Since n=30, we can just loop
                verts = [i for i in range(n) if (mask >> i) & 1]
                if len(verts) < 4:
                    continue
                
                # Check for K2,2 in verts
                # A K2,2 is {x,y} and {z,w} such that x,z; x,w; y,z; y,w are edges.
                # This is equivalent to: for some x,y, |N(x) \cap N(y) \cap verts| >= 2.
                for i_idx in range(len(verts)):
                    x = verts[i_idx]
                    for j_idx in range(i_idx + 1, len(verts)):
                        y = verts[j_idx]
                        # Common neighbors of x and y within verts
                        common_mask = neighbors[x] & neighbors[y] & mask
                        # Count bits
                        cnt = bin(common_mask).count('1')
                        if cnt >= 2:
                            # Each pair of common neighbors forms a K2,2 with {x,y}
                            # Number of pairs is C(cnt, 2)
                            count += cnt * (cnt - 1) // 2
        return count

    def total_violations(colors):
        adj0, adj1 = build_adjacency(colors)
        return count_k222(adj0) + count_k222(adj1)

    best_colors = None
    best_score = float('inf')
    
    # Try multiple random restarts
    while time.time() < deadline:
        colors = np.random.randint(0, 2, n_edges).tolist()
        
        # Greedy local search
        current_score = total_violations(colors)
        
        # If we start with 0, we are done
        if current_score == 0:
            return {'n': n, 'colors': colors}
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Pick random edge to flip
            edge_idx = np.random.randint(0, n_edges)
            old_val = colors[edge_idx]
            colors[edge_idx] = 1 - old_val
            
            new_score = total_violations(colors)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score == 0:
                    return {'n': n, 'colors': colors}
            else:
                # Revert
                colors[edge_idx] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
        # If we found a valid one in the middle, we would have returned.
        # If we are stuck in a local minimum, restart.
        
    # If we didn't find n=30, try n=29?
    # But n=30 is known to exist. The search should find it.
    # If time runs out, return best found. If best_score > 0, it's invalid.
    # We must return a valid coloring.
    # If we failed to find valid n=30, we should fallback to a known valid smaller n.
    # But the goal is to maximize n.
    # If we can't find valid 30, we return valid 29?
    # We need a valid coloring to return.
    # Let's ensure we have a valid coloring for n=17 (Paley) as a fallback.
    
    # Fallback: n=17 Paley
    squares = {x * x % 17 for x in range(1, 17)}
    fallback_colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    
    if best_score == 0:
        return {'n': n, 'colors': best_colors}
    
    # If we didn't find a valid 30, maybe we found a valid 29?
    # The search above only ran for n=30.
    # We should probably just return the fallback if we fail, to ensure validity.
    # But the score is n. Returning 17 is bad if 29 is possible.
    # However, if the search for 30 fails, it's likely due to time/algorithm limits.
    # Given R(K2,2,2)=31, 30 is the max.
    # If we don't find 30, we score 0 or low.
    # To be safe, if best_score > 0, we return the fallback valid coloring.
    
    return {'n': 17, 'colors': fallback_colors}

# EVOLVE-BLOCK-END
