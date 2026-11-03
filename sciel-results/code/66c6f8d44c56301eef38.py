"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s wall-clock limit with safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We will construct a decomposition using recursive Strassen.
    # For 4x4 * 4x6, we can view it as 4x4 * (4x4 + 4x2)
    # But it's easier to just use the standard recursive Strassen for 4x4 * 4x6?
    # Strassen works for any n x m * m x p where n, m, p are powers of 2?
    # No, Strassen is for square matrices or can be padded.
    # However, we can just use the schoolbook for the 4x2 part and Strassen for 4x4 part?
    # Or better, just use the full Strassen recursion for 4x4 * 4x6 by padding B to 4x8?
    # No, that increases rank.
    
    # Let's try to implement a general recursive Strassen for n x m * m x p.
    # If n, m, p are not all powers of 2, we can still apply the 2x2 block decomposition
    # if n, m, p are even.
    
    # n=4, m=4, p=6.
    # n=4 (even), m=4 (even), p=6 (even).
    # So we can split A into 2x2 blocks of 2x4? No, standard Strassen splits into 2x2 blocks of n/2 x m/2.
    # A: 4x4 -> 2x2 blocks of 2x2.
    # B: 4x6 -> 2x2 blocks of 2x3.
    # C: 4x6 -> 2x2 blocks of 2x3.
    
    # So we need Strassen for 2x2 * 2x3.
    # And for the block multiplications, we need 2x2 * 2x3.
    # And for the base case, we need 1x1 * 1x?
    
    # Let's define a function to build the decomposition for n x m * m x p.
    
    def build_decomp(n, m, p):
        # Returns U, V, W for n x m * m x p
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Base case: 1x1 * 1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # If any dimension is odd, we can't use standard 2x2 Strassen directly without padding.
        # But here n=4, m=4, p=6 are all even.
        # So we can split.
        
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
            # Fallback to schoolbook for simplicity in this specific case, 
            # but we expect n,m,p to be even in our recursion.
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return U, V, W
        
        # Recursive case
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        # We need to combine 7 sub-decompositions.
        # The sub-problems are:
        # 1. (A11+A22) * (B11+B22) -> n2 x m2 * m2 x p2
        # 2. (A21+A22) * B11 -> n2 x m2 * m2 x p2
        # 3. A11 * (B12-B22) -> n2 x m2 * m2 x p2
        # 4. A22 * (B21-B11) -> n2 x m2 * m2 x p2
        # 5. (A11+A12) * B22 -> n2 x m2 * m2 x p2
        # 6. (A21-A11) * (B11+B12) -> n2 x m2 * m2 x p2
        # 7. (A12-A22) * (B21+B22) -> n2 x m2 * m2 x p2
        
        # We get the decompositions for these 7 problems.
        # Note: The dimensions of the sub-problems are all n2 x m2 * m2 x p2.
        # So we can just call build_decomp(n2, m2, p2) seven times?
        # Yes, because the "matrices" in the sub-problems are linear combinations of the blocks.
        
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        
        # We need to map the indices.
        # A is n x m.
        # A11: rows 0..n2-1, cols 0..m2-1. Index in U: i*m + j.
        # A12: rows 0..n2-1, cols m2..m-1. Index in U: i*m + (j+m2).
        # A21: rows n2..n-1, cols 0..m2-1. Index in U: (i+n2)*m + j.
        # A22: rows n2..n-1, cols m2..m-1. Index in U: (i+n2)*m + (j+m2).
        
        # Similarly for B (m x p) and W (p x n).
        # B11: rows 0..m2-1, cols 0..p2-1. Index in V: j*p + k.
        # B12: rows 0..m2-1, cols p2..p-1. Index in V: j*p + (k+p2).
        # B21: rows m2..m-1, cols 0..p2-1. Index in V: (j+m2)*p + k.
        # B22: rows m2..m-1, cols p2..p-1. Index in V: (j+m2)*p + (k+p2).
        
        # W (C is n x p):
        # C11: rows 0..n2-1, cols 0..p2-1. Index in W: k*n + i.
        # C12: rows 0..n2-1, cols p2..p-1. Index in W: (k+p2)*n + i.
        # C21: rows n2..n-1, cols 0..p2-1. Index in W: k*n + (i+n2).
        # C22: rows n2..n-1, cols p2..p-1. Index in W: (k+p2)*n + (i+n2).
        
        # The sub-decomposition U_sub has shape (n2*m2, rank_sub).
        # We need to map each element of U_sub (which corresponds to a local index i_local, j_local)
        # to the global indices for the linear combinations.
        
        # Let's define a helper to get the global index in U, V, W for a local index in the sub-problem.
        # Local indices in sub-problem: i in 0..n2-1, j in 0..m2-1, k in 0..p2-1.
        # U_local index: i*m2 + j.
        # V_local index: j*p2 + k.
        # W_local index: k*n2 + i.
        
        # For a given term in U_sub at local index (i, j), we need to add it to the global U
        # at the positions corresponding to the linear combination.
        
        # Let's create the full U, V, W arrays.
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Helper to add a term from sub-decomp to global
        def add_sub_term(global_U, global_V, global_W, 
                         u_local_idx, v_local_idx, w_local_idx, 
                         u_map, v_map, w_map, 
                         coeff, base_col):
            # u_map, v_map, w_map are lists of (global_idx, coeff)
            # We add coeff * U_sub[u_local_idx, c] to global_U[g_u, base_col + c]
            # etc.
            for c in range(rank_sub):
                val = coeff * U_sub[u_local_idx][c]
                if val != 0:
                    for g_u, cu in u_map:
                        global_U[g_u][base_col + c] += val * cu
                val = coeff * V_sub[v_local_idx][c]
                if val != 0:
                    for g_v, cv in v_map:
                        global_V[g_v][base_col + c] += val * cv
                val = coeff * W_sub[w_local_idx][c]
                if val != 0:
                    for g_w, cw in w_map:
                        global_W[g_w][base_col + c] += val * cw
        
        # We need to iterate over all local indices in the sub-problem.
        # There are n2*m2 local U indices, m2*p2 local V indices, p2*n2 local W indices.
        # But they are coupled by the rank dimension.
        # Actually, the sub-decomp is a set of rank_sub terms.
        # Each term c in 0..rank_sub-1 has a specific pattern in U_sub, V_sub, W_sub.
        # We can just iterate over c and the local indices?
        # No, it's easier to iterate over the local indices and for each, determine the mapping.
        
        # Let's precompute the mappings for each of the 7 Strassen terms.
        # Term 1: (A11+A22)(B11+B22)
        # A part: A11 (+1), A22 (+1)
        # B part: B11 (+1), B22 (+1)
        # C part: C11 (+1), C22 (+1)
        
        # For a local index (i, j) in A11, global U index is i*m + j.
        # For a local index (i, j) in A22, global U index is (i+n2)*m + (j+m2).
        
        # So for a given local U index u_loc = i*m2 + j:
        # i = u_loc // m2, j = u_loc % m2.
        # u_map for Term 1: [(i*m + j, 1), ((i+n2)*m + (j+m2), 1)]
        
        # Similarly for V and W.
        
        # Let's define the 7 terms' mappings.
        # Each term is a tuple of (u_map_func, v_map_func, w_map_func)
        # where u_map_func(i, j) returns list of (global_idx, coeff)
        
        def get_u_map(i, j, block):
            # block: 11, 12, 21, 22
            if block == 11:
                return [(i*m + j, 1)]
            elif block == 12:
                return [(i*m + (j+m2), 1)]
            elif block == 21:
                return [((i+n2)*m + j, 1)]
            elif block == 22:
                return [((i+n2)*m + (j+m2), 1)]
        
        def get_v_map(j, k, block):
            if block == 11:
                return [(j*p + k, 1)]
            elif block == 12:
                return [(j*p + (k+p2), 1)]
            elif block == 21:
                return [((j+m2)*p + k, 1)]
            elif block == 22:
                return [((j+m2)*p + (k+p2), 1)]
        
        def get_w_map(k, i, block):
            if block == 11:
                return [(k*n + i, 1)]
            elif block == 12:
                return [((k+p2)*n + i, 1)]
            elif block == 21:
                return [(k*n + (i+n2), 1)]
            elif block == 22:
                return [((k+p2)*n + (i+n2), 1)]
        
        # Term 1: A: 11, 22. B: 11, 22. C: 11, 22.
        # Term 2: A: 21, 22. B: 11. C: 21, 22.
        # Term 3: A: 11. B: 12, 22. C: 12, 22.
        # Term 4: A: 22. B: 21, 11. C: 11, 21.
        # Term 5: A: 11, 12. B: 22. C: 11, 12.
        # Term 6: A: 21, 11. B: 11, 12. C: 22.
        # Term 7: A: 12, 22. B: 21, 22. C: 11.
        
        # Note: The signs are important.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Wait, the W mapping depends on which C block the term contributes to.
        # So for each of the 7 terms, we need to specify the W mapping.
        
        # Term 1 (M1):
        # A: 11 (+1), 22 (+1)
        # B: 11 (+1), 22 (+1)
        # C: 11 (+1), 22 (+1)
        
        # Term 2 (M2):
        # A: 21 (+1), 22 (+1)
        # B: 11 (+1)
        # C: 21 (+1), 22 (-1)
        
        # Term 3 (M3):
        # A: 11 (+1)
        # B: 12 (+1), 22 (-1)
        # C: 12 (+1), 22 (+1)
        
        # Term 4 (M4):
        # A: 22 (+1)
        # B: 21 (+1), 11 (-1)
        # C: 11 (+1), 21 (+1)
        
        # Term 5 (M5):
        # A: 11 (+1), 12 (+1)
        # B: 22 (+1)
        # C: 11 (-1), 12 (+1)
        
        # Term 6 (M6):
        # A: 21 (+1), 11 (-1)
        # B: 11 (+1), 12 (+1)
        # C: 22 (+1)
        
        # Term 7 (M7):
        # A: 12 (+1), 22 (-1)
        # B: 21 (+1), 22 (+1)
        # C: 11 (+1)
        
        # Let's encode these.
        # Each term is a dict with 'A', 'B', 'C' keys.
        # Value is a list of (block, coeff).
        
        terms = [
            {'A': [(11, 1), (22, 1)], 'B': [(11, 1), (22, 1)], 'C': [(11, 1), (22, 1)]},
            {'A': [(21, 1), (22, 1)], 'B': [(11, 1)], 'C': [(21, 1), (22, -1)]},
            {'A': [(11, 1)], 'B': [(12, 1), (22, -1)], 'C': [(12, 1), (22, 1)]},
            {'A': [(22, 1)], 'B': [(21, 1), (11, -1)], 'C': [(11, 1), (21, 1)]},
            {'A': [(11, 1), (12, 1)], 'B': [(22, 1)], 'C': [(11, -1), (12, 1)]},
            {'A': [(21, 1), (11, -1)], 'B': [(11, 1), (12, 1)], 'C': [(22, 1)]},
            {'A': [(12, 1), (22, -1)], 'B': [(21, 1), (22, 1)], 'C': [(11, 1)]}
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            
            # Iterate over all local indices
            for i in range(n2):
                for j in range(m2):
                    u_loc = i * m2 + j
                    if all(U_sub[u_loc][c] == 0 for c in range(rank_sub)):
                        continue
                    
                    # Get U mapping
                    u_map = []
                    for block, coeff in term['A']:
                        # Only if the local index (i,j) is in this block?
                        # No, the local index (i,j) is always in the sub-problem.
                        # The block determines the global position.
                        # So for block 11, global idx is i*m + j.
                        # For block 22, global idx is (i+n2)*m + (j+m2).
                        # We add (global_idx, coeff).
                        if block == 11:
                            u_map.append((i*m + j, coeff))
                        elif block == 12:
                            u_map.append((i*m + (j+m2), coeff))
                        elif block == 21:
                            u_map.append(((i+n2)*m + j, coeff))
                        elif block == 22:
                            u_map.append(((i+n2)*m + (j+m2), coeff))
                    
                    # Now we need to find the corresponding v_loc and w_loc.
                    # This is tricky because the sub-decomp couples i, j, k.
                    # We can't just iterate i, j independently.
                    # We should iterate over the rank dimension c, and for each c,
                    # find the non-zero entries in U_sub, V_sub, W_sub.
                    
                    # Actually, it's easier to iterate over c.
                    pass
        
        # Let's restart the combination loop.
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            
            for c in range(rank_sub):
                # Find non-zero entries in U_sub[:, c], V_sub[:, c], W_sub[:, c]
                u_nonzero = [(idx, val) for idx, val in enumerate(U_sub) if val[c] != 0]
                v_nonzero = [(idx, val) for idx, val in enumerate(V_sub) if val[c] != 0]
                w_nonzero = [(idx, val) for idx, val in enumerate(W_sub) if val[c] != 0]
                
                # For each non-zero entry, we need to map it to global indices.
                # But the term defines the mapping for the whole sub-problem.
                # So for a local u_idx, we determine the global u indices based on term['A'].
                
                for u_idx, u_val in u_nonzero:
                    i = u_idx // m2
                    j = u_idx % m2
                    for block, coeff in term['A']:
                        if block == 11:
                            g_u = i*m + j
                        elif block == 12:
                            g_u = i*m + (j+m2)
                        elif block == 21:
                            g_u = (i+n2)*m + j
                        elif block == 22:
                            g_u = (i+n2)*m + (j+m2)
                        else:
                            continue
                        U[g_u][base_col + c] += u_val[c] * coeff
                
                for v_idx, v_val in v_nonzero:
                    j = v_idx // p2
                    k = v_idx % p2
                    for block, coeff in term['B']:
                        if block == 11:
                            g_v = j*p + k
                        elif block == 12:
                            g_v = j*p + (k+p2)
                        elif block == 21:
                            g_v = (j+m2)*p + k
                        elif block == 22:
                            g_v = (j+m2)*p + (k+p2)
                        else:
                            continue
                        V[g_v][base_col + c] += v_val[c] * coeff
                
                for w_idx, w_val in w_nonzero:
                    k = w_idx // n2
                    i = w_idx % n2
                    for block, coeff in term['C']:
                        if block == 11:
                            g_w = k*n + i
                        elif block == 12:
                            g_w = (k+p2)*n + i
                        elif block == 21:
                            g_w = k*n + (i+n2)
                        elif block == 22:
                            g_w = (k+p2)*n + (i+n2)
                        else:
                            continue
                        W[g_w][base_col + c] += w_val[c] * coeff
        
        return U, V, W
    
    U, V, W = build_decomp(n, m, p)
    
    # Verify
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
