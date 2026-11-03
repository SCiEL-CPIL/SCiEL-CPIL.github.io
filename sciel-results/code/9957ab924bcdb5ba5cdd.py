"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def build_decomp(n, m, p):
        """Build tensor decomposition for n x m * m x p matrix multiplication.
        Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) as lists.
        """
        # Base case: 1x1 * 1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # Special case: 2x2 * 2x3 (or permutations)
        # Rank 11 decomposition: 7 (Strassen for 2x2x2) + 4 (Schoolbook for 2x2x1)
        if n == 2 and m == 2 and p == 3:
            rank = 11
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Strassen terms for cols 0,1
            # U indices: 0,1,2,3 (A00, A01, A10, A11)
            # V indices: 0,1,3,4 (B00, B01, B10, B11) - note p=3, so B00=0, B01=1, B10=3, B11=4
            # W indices: 0,1,2,3 (C00, C01, C10, C11)
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            terms_strassen = [
                # U coeffs, V coeffs, W coeffs
                ([(0, 1), (3, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)]),
                ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
                ([(0, 1)], [(1, 1), (4, -1)], [(1, 1), (3, 1)]),
                ([(3, 1)], [(3, 1), (0, -1)], [(0, 1), (2, 1)]),
                ([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (1, 1)]),
                ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
                ([(1, 1), (3, -1)], [(3, 1), (4, 1)], [(0, 1)])
            ]
            
            for t_idx, (u_map, v_map, w_map) in enumerate(terms_strassen):
                for idx, coeff in u_map:
                    U[idx][t_idx] += coeff
                for idx, coeff in v_map:
                    V[idx][t_idx] += coeff
                for idx, coeff in w_map:
                    W[idx][t_idx] += coeff
            
            # Schoolbook for col 2
            # C02 = A00 B02 + A01 B12
            # C12 = A10 B02 + A11 B12
            # V indices: B02=2, B12=5
            # W indices: C02=4, C12=5
            
            # Term 7: A00 B02 -> C02
            U[0][7] = 1; V[2][7] = 1; W[4][7] = 1
            # Term 8: A01 B12 -> C02
            U[1][8] = 1; V[5][8] = 1; W[4][8] = 1
            # Term 9: A10 B02 -> C12
            U[2][9] = 1; V[2][9] = 1; W[5][9] = 1
            # Term 10: A11 B12 -> C12
            U[3][10] = 1; V[5][10] = 1; W[5][10] = 1
            
            return U, V, W

        # If any dimension is odd, fall back to schoolbook (or other specific cases)
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
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
            return U, V, w
        
        # Recursive Strassen case
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
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
            
            for c in range(rank_sub):
                for u_idx in range(n2 * m2):
                    u_val = U_sub[u_idx][c]
                    if u_val == 0: continue
                    i = u_idx // m2
                    j = u_idx % m2
                    for block, coeff in term['A']:
                        if block == 11: g_u = i * m + j
                        elif block == 12: g_u = i * m + (j + m2)
                        elif block == 21: g_u = (i + n2) * m + j
                        elif block == 22: g_u = (i + n2) * m + (j + m2)
                        else: continue
                        U[g_u][base_col + c] += u_val * coeff
                
                for v_idx in range(m2 * p2):
                    v_val = V_sub[v_idx][c]
                    if v_val == 0: continue
                    j = v_idx // p2
                    k = v_idx % p2
                    for block, coeff in term['B']:
                        if block == 11: g_v = j * p + k
                        elif block == 12: g_v = j * p + (k + p2)
                        elif block == 21: g_v = (j + m2) * p + k
                        elif block == 22: g_v = (j + m2) * p + (k + p2)
                        else: continue
                        V[g_v][base_col + c] += v_val * coeff
                
                for w_idx in range(p2 * n2):
                    w_val = W_sub[w_idx][c]
                    if w_val == 0: continue
                    k = w_idx // n2
                    i = w_idx % n2
                    for block, coeff in term['C']:
                        if block == 11: g_w = k * n + i
                        elif block == 12: g_w = (k + p2) * n + i
                        elif block == 21: g_w = k * n + (i + n2)
                        elif block == 22: g_w = (k + p2) * n + (i + n2)
                        else: continue
                        W[g_w][base_col + c] += w_val * coeff
        
        return U, V, W
    
    U, V, W = build_decomp(n, m, p)
    
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
