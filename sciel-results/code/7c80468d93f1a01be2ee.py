import time
import numpy as np
import random

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    def count_k222(adj, n):
        """
        Counts the number of K2,2,2 subgraphs in the graph defined by adj (boolean matrix).
        Uses the property: K2,2,2 exists if there are u, v such that N(u) intersect N(v) contains a K2,2.
        """
        # Precompute neighbor masks for faster intersection
        # n <= 30, so we can use Python integers as bitmasks
        neighbors = [0] * n
        for i in range(n):
            mask = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    mask |= (1 << j)
            neighbors[i] = mask
        
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                # S = N(u) & N(v)
                s_mask = neighbors[u] & neighbors[v]
                # Remove u and v from S? 
                # u is not in N(u). v is in N(u) if edge(u,v) exists.
                # In K2,2,2, the 6 vertices are distinct.
                # So u and v must not be in S?
                # Actually, if u and v are in A, then B and C are in N(u) and N(v).
                # The vertices in B and C must be distinct from u and v.
                # So we should remove u and v from the candidate set for B and C.
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v)
                
                if bin(s_mask).count('1') < 4:
                    continue
                
                # Extract indices in S
                # Since n is small, we can just iterate
                verts = []
                for i in range(n):
                    if (s_mask >> i) & 1:
                        verts.append(i)
                
                # Check for K2,2 in the subgraph induced by verts (using edges from adj)
                # K2,2 exists if there is a pair (b1, b2) in verts such that 
                # |N(b1) & N(b2) & S| >= 2
                l = len(verts)
                for i in range(l):
                    b1 = verts[i]
                    for j in range(i + 1, l):
                        b2 = verts[j]
                        # Common neighbors of b1 and b2 within S
                        common = neighbors[b1] & neighbors[b2] & s_mask
                        if bin(common).count('1') >= 2:
                            # Each pair of common neighbors forms a K2,2 with {b1, b2}
                            # Number of pairs is C(k, 2)
                            k = bin(common).count('1')
                            count += k * (k - 1) // 2
        return count

    def total_violations(colors, n):
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
        
        return count_k222(adj0, n) + count_k222(adj1, n)

    # 1. Generate Paley(29) coloring
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            # Edge (i,j) is in Paley graph if j-i is a square
            # We map Paley edges to color 0, others to 1
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
    
    # Verify n=29
    v29 = total_violations(colors_29, 29)
    if v29 == 0:
        best_n = 29
        best_colors = colors_29
    else:
        # Fallback to smaller n if Paley(29) is invalid (unlikely)
        best_n = 1
        best_colors = [0]
        
    # 2. Try to find n=30
    n = 30
    n_edges = n * (n - 1) // 2
    # Start with Paley(29) extended with 1 vertex
    # The first 29*28/2 = 406 edges are from Paley(29)
    # The remaining 29 edges connect vertex 29 to 0..28
    init_colors = colors_29.copy()
    # Initialize new edges randomly
    for _ in range(29):
        init_colors.append(random.randint(0, 1))
    
    current_colors = init_colors
    current_violations = total_violations(current_colors, n)
    
    if current_violations == 0:
        return {'n': n, 'colors': current_colors}
        
    # Local Search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Pick random edge to flip
        edge_idx = random.randint(0, n_edges - 1)
        old_val = current_colors[edge_idx]
        current_colors[edge_idx] = 1 - old_val
        
        new_violations = total_violations(current_colors, n)
        
        if new_violations < current_violations:
            current_violations = new_violations
            improved = True
            if current_violations == 0:
                return {'n': n, 'colors': current_colors}
        else:
            # Revert
            current_colors[edge_idx] = old_val
            
    # If we found a valid n=30, we returned. 
    # If not, we return the best valid one we have (n=29)
    # Note: It's possible the search improved the n=30 coloring but didn't reach 0 violations.
    # In that case, it's invalid, so we must return the valid n=29.
    
    return {'n': best_n, 'colors': best_colors}
