import numpy as np
import time

def construct():
    n = 29
    # Paley graph on Z_29
    residues = set()
    for x in range(1, 29):
        residues.add((x * x) % 29)
    
    # Build adjacency matrix for colors 0 and 1 for K_29
    # adj[0][i][j] = 1 if color(i,j) is 0
    # adj[1][i][j] = 1 if color(i,j) is 1
    adj = np.zeros((2, 29, 29), dtype=np.int8)
    for i in range(29):
        for j in range(i + 1, 29):
            diff = (j - i) % 29
            if diff in residues:
                adj[0][i][j] = 1
                adj[0][j][i] = 1
            else:
                adj[1][i][j] = 1
                adj[1][j][i] = 1
                
    colors_29 = []
    for i in range(29):
        for j in range(i + 1, 29):
            colors_29.append(1 if adj[1][i][j] else 0)

    # Try to extend to n=30
    start_time = time.time()
    deadline = start_time + 100.0 # Keep 20s margin
    
    # Precompute C4 detection helper?
    # For a set of vertices U and a color k, check if subgraph G_k[U] has a C4.
    # G_k[U] has edge (u,v) if adj[k][u][v] == 1.
    # C4 exists if there are u,v,w,x in U such that (u,v), (v,w), (w,x), (x,u) are edges.
    # Or simply, for any pair (u,v) in U, if they share 2 common neighbors in U, that's a C4?
    # No, common neighbors form a path u-w-v and u-x-v. If w!=x, then u-w-v-x-u is a C4?
    # Edges: (u,w), (w,v), (v,x), (x,u). Yes.
    # So for each pair (u,v) in U, count common neighbors in U.
    # If count >= 2, there is a C4.
    
    def has_c4_in_subset(U, k):
        # U is a list of vertex indices
        if len(U) < 4:
            return False
        # Map U to indices 0..m-1 for faster access?
        # Or just use the global adj matrix.
        m = len(U)
        # Common neighbors of u and v in U
        for i in range(m):
            u = U[i]
            for j in range(i + 1, m):
                v = U[j]
                if adj[k][u][v] == 0:
                    continue # u and v not connected, can't be part of C4 as opposite? 
                    # Wait, in C4 u-w-v-x-u, u and v are opposite. They are NOT connected.
                    # So we check pairs (u,v) that are NOT connected.
                    # If they have >= 2 common neighbors in U, then C4 exists.
                    # Let's refine:
                    # For C4, we need 4 distinct vertices a,b,c,d.
                    # Edges (a,b), (b,c), (c,d), (d,a).
                    # a and c are opposite. b and d are opposite.
                    # So for each pair (a,c) in U that is NOT an edge in G_k:
                    #   Find common neighbors N = { x in U | adj[k][a][x]==1 and adj[k][c][x]==1 }
                    #   If |N| >= 2, pick b, d from N. Then a-b-c-d-a is a C4.
                    #   Wait, we also need edges (b,d)? No, C4 is a cycle.
                    #   The cycle is a-b-c-d-a.
                    #   Edges required: (a,b), (b,c), (c,d), (d,a).
                    #   b and d are in N, so (a,b), (b,c), (a,d), (d,c) exist.
                    #   This forms a "diamond" or two triangles sharing base ac? No.
                    #   It forms a C4 if b != d.
                    #   Yes, a-b-c-d-a is a simple cycle of length 4.
                    #   So if there are 2 distinct common neighbors, we have a C4.
                    
                    common = []
                    for x in U:
                        if x != u and x != v:
                            if adj[k][u][x] == 1 and adj[k][v][x] == 1:
                                common.append(x)
                                if len(common) >= 2:
                                    return True
        return False

    # This is O(m^3). m <= 28. 28^3 = 21952.
    # We call this for each a2 in Vk.
    # Vk size ~ 14. U size ~ 14.
    # 29 * 2 * 14^3 ~ 160,000 ops per evaluation.
    # 100,000 evaluations in 100s? 1000/s.
    # 160,000 * 1000 = 160,000,000 ops. Might be slow in pure Python.
    # Let's optimize has_c4_in_subset.
    
    def has_c4_opt(U, k):
        m = len(U)
        if m < 4: return False
        # Bitmask representation of U?
        # Or just numpy.
        # adj_k = adj[k]
        # Submatrix S = adj_k[np.ix_(U, U)]
        # S is m x m.
        # We want to know if S contains a C4.
        # C4 exists if (S @ S.T) has an entry >= 2 for a non-adjacent pair?
        # S @ S.T gives number of common neighbors.
        # For pair (i,j) in U, if S[i,j]==0 and (S@S.T)[i,j] >= 2, then C4.
        
        S = adj[k][np.ix_(U, U)]
        # S is int8.
        # Compute common neighbors
        # Use matrix multiplication
        # S is 0/1.
        # C = S @ S.T
        C = np.dot(S, S.T)
        # Check off-diagonal elements where S[i,j] == 0
        # Mask where S == 0 and i != j
        # np.fill_diagonal(S, 1) to ignore self? No, we want i!=j.
        # Create mask: ~S & (1 - np.eye(m, dtype=int8))
        mask = (S == 0)
        np.fill_diagonal(mask, False)
        
        # If any C[i,j] >= 2 where mask[i,j] is True, return True
        # C is int.
        # We can just check:
        return np.any(C[mask] >= 2)
        
    # This is much faster.
    
    def evaluate(c):
        # c is array of 29 values (0 or 1)
        violations = 0
        for k in [0, 1]:
            Vk = [i for i in range(29) if c[i] == k]
            for a2 in Vk:
                # U = neighbors of a2 in Vk with color k
                # adj[k][a2] is a vector of 29.
                # We want j in Vk such that j != a2 and adj[k][a2][j] == 1
                # Vk is a list.
                # Use numpy for speed
                Vk_arr = np.array(Vk)
                # Filter Vk_arr
                # adj_k_a2 = adj[k][a2]
                # mask = adj_k_a2[Vk_arr] == 1
                # Also exclude a2
                mask = (adj[k][a2][Vk_arr] == 1) & (Vk_arr != a2)
                U = Vk_arr[mask]
                if len(U) >= 4:
                    if has_c4_opt(U, k):
                        violations += 1
        return violations

    best_c = None
    best_violations = float('inf')
    
    # Random restarts
    rng = np.random.default_rng(42)
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        c = rng.integers(0, 2, 29)
        v = evaluate(c)
        if v == 0:
            best_c = c
            break
        if v < best_violations:
            best_violations = v
            best_c = c.copy()
            
        # Local search
        for step in range(100):
            if time.time() > deadline:
                break
            if v == 0:
                best_c = c
                break
            # Pick a random vertex to flip
            idx = rng.integers(0, 29)
            c[idx] = 1 - c[idx]
            v_new = evaluate(c)
            if v_new <= v:
                v = v_new
                if v < best_violations:
                    best_violations = v
                    best_c = c.copy()
            else:
                c[idx] = 1 - c[idx] # Revert
                
    if best_c is not None and evaluate(best_c) == 0:
        # Construct n=30 coloring
        colors_30 = colors_29.copy()
        # Add edges from 29 to 0..28
        # Lexicographic order: (0,1), (0,2), ..., (0,28), (1,2), ...
        # New edges are (i, 29) for i=0..28.
        # These come at the end of the list?
        # Pairs (i,j) with i<j.
        # Max pair is (28, 29).
        # The list is generated by:
        # for i in range(n): for j in range(i+1, n)
        # For n=30, the last 29 edges are (0,29), (1,29), ..., (28,29).
        # So we just append best_c.
        colors_30.extend(best_c.tolist())
        return {'n': 30, 'colors': colors_30}
    else:
        return {'n': 29, 'colors': colors_29}
