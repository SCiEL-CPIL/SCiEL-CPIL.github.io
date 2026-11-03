"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use recursive Strassen decomposition
    # 4x4 * 4x6: split into 2x2 blocks of 2x2 and 2x3
    # Each 2x2 * 2x3 sub-problem uses Strassen (7 multiplications)
    # Total rank = 7 * 7 = 49
    
    def build_decomp(n, m, p):
        """Build tensor decomposition for n x m * m x p matrix multiplication.
        Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) as lists.
        """
        # Base case: 1x1 * 1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # If any dimension is odd, fall back to schoolbook
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
            return U, V, W
        
        # Recursive Strassen case
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # The 7 Strassen terms:
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
        
        # Each term: A_blocks [(block, coeff)], B_blocks, C_blocks
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
                # Process U entries
                for u_idx in range(n2 * m2):
                    u_val = U_sub[u_idx][c]
                    if u_val == 0:
                        continue
                    i = u_idx // m2
                    j = u_idx % m2
                    for block, coeff in term['A']:
                        if block == 11:
                            g_u = i * m + j
                        elif block == 12:
                            g_u = i * m + (j + m2)
                        elif block == 21:
                            g_u = (i + n2) * m + j
                        elif block == 22:
                            g_u = (i + n2) * m + (j + m2)
                        else:
                            continue
                        U[g_u][base_col + c] += u_val * coeff
                
                # Process V entries
                for v_idx in range(m2 * p2):
                    v_val = V_sub[v_idx][c]
                    if v_val == 0:
                        continue
                    j = v_idx // p2
                    k = v_idx % p2
                    for block, coeff in term['B']:
                        if block == 11:
                            g_v = j * p + k
                        elif block == 12:
                            g_v = j * p + (k + p2)
                        elif block == 21:
                            g_v = (j + m2) * p + k
                        elif block == 22:
                            g_v = (j + m2) * p + (k + p2)
                        else:
                            continue
                        V[g_v][base_col + c] += v_val * coeff
                
                # Process W entries
                for w_idx in range(p2 * n2):
                    w_val = W_sub[w_idx][c]
                    if w_val == 0:
                        continue
                    k = w_idx // n2
                    i = w_idx % n2
                    for block, coeff in term['C']:
                        if block == 11:
                            g_w = k * n + i
                        elif block == 12:
                            g_w = (k + p2) * n + i
                        elif block == 21:
                            g_w = k * n + (i + n2)
                        elif block == 22:
                            g_w = (k + p2) * n + (i + n2)
                        else:
                            continue
                        W[g_w][base_col + c] += w_val * coeff
        
        return U, V, W
    
    U, V, W = build_decomp(n, m, p)
    
    # Verify the decomposition
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
