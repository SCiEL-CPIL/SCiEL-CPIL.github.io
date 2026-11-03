import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s wall-clock limit with safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def build_decomp(n, m, p):
        # Returns U, V, W for n x m * m x p
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Base case: 1x1 * 1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # If any dimension is odd, we can't use standard 2x2 Strassen directly without padding.
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
            # Fallback to schoolbook
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
                u_nonzero = [(idx, val) for idx, val in enumerate(U_sub) if val[c] != 0]
                v_nonzero = [(idx, val) for idx, val in enumerate(V_sub) if val[c] != 0]
                w_nonzero = [(idx, val) for idx, val in enumerate(W_sub) if val[c] != 0]
                
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

    # Construct 4x4x4 and 4x4x2 decompositions
    U1, V1, W1 = build_decomp(4, 4, 4)
    U2, V2, W2 = build_decomp(4, 4, 2)
    
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(24)]
    W = [[0] * total_rank for _ in range(24)]
    
    # Copy U1, V1, W1
    for i in range(16):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    
    for i in range(16):
        for c in range(rank1):
            V[i][c] = V1[i][c]
            
    for i in range(16):
        for c in range(rank1):
            W[i][c] = W1[i][c]
            
    # Copy U2, V2, W2 with offset
    for i in range(16):
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
            
    # Map V2 indices (j*2 + k) to global (j*6 + (4+k))
    for i in range(8):
        j = i // 2
        k = i % 2
        g_v = j * 6 + (4 + k)
        for c in range(rank2):
            V[g_v][rank1 + c] = V2[i][c]
            
    # Map W2 indices (k*4 + i) to global ((4+k)*4 + i)
    for i in range(8):
        k = i // 4
        r = i % 4
        g_w = (4 + k) * 4 + r
        for c in range(rank2):
            W[g_w][rank1 + c] = W2[i][c]
            
    return {"U": U, "V": V, "W": W}
