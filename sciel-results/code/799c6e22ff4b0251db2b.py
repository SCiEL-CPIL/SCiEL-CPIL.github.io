import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    
    # Target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Try BCLR 23-rank for 3x3 blocks inside Strassen 2x2
    # Rank = 7 * 23 = 161
    
    def get_bclr_3x3():
        # Returns U, V, W for 3x3 multiplication, rank 23
        # Indices: A (0..8), B (0..8), C (0..8)
        # A_ij = A[i*3+j], B_kj = B[k*3+j], C_ij = C[i*3+j]
        U = np.zeros((9, 23), dtype=float)
        V = np.zeros((9, 23), dtype=float)
        W = np.zeros((9, 23), dtype=float)
        
        # The 23 products
        # Format: (A_indices, coeffs), (B_indices, coeffs), (C_indices, coeffs)
        # A_indices are (i, k) for A[i, k]
        # B_indices are (k, j) for B[k, j]
        # C_indices are (i, j) for C[i, j]
        
        def idx(a, b, size=3):
            return a * size + b

        terms = []
        
        # 1-9: Diagonal-like
        terms.append(([(0,0)], [1.0], [(0,0)], [1.0], [(0,0)], [1.0])) # A00 B00 -> C00
        terms.append(([(0,1)], [1.0], [(1,1)], [1.0], [(0,1)], [1.0])) # A01 B11 -> C01
        terms.append(([(0,2)], [1.0], [(2,0)], [1.0], [(0,2)], [1.0])) # A02 B20 -> C02
        terms.append(([(1,0)], [1.0], [(0,1)], [1.0], [(1,0)], [1.0])) # A10 B01 -> C10
        terms.append(([(1,1)], [1.0], [(1,0)], [1.0], [(1,1)], [1.0])) # A11 B10 -> C11
        terms.append(([(1,2)], [1.0], [(2,1)], [1.0], [(1,2)], [1.0])) # A12 B21 -> C12
        terms.append(([(2,0)], [1.0], [(1,2)], [1.0], [(2,0)], [1.0])) # A20 B12 -> C20
        terms.append(([(2,1)], [1.0], [(2,2)], [1.0], [(2,1)], [1.0])) # A21 B22 -> C21
        terms.append(([(2,2)], [1.0], [(0,2)], [1.0], [(2,2)], [1.0])) # A22 B02 -> C22
        
        # 10-18: Pairs
        terms.append(([(0,0),(1,1)], [1.0,1.0], [(0,0),(1,1)], [1.0,1.0], [(0,0),(1,1)], [1.0,1.0]))
        terms.append(([(0,1),(1,2)], [1.0,1.0], [(1,0),(2,1)], [1.0,1.0], [(0,2),(1,1),(2,0)], [1.0,1.0,1.0]))
        terms.append(([(0,2),(1,0)], [1.0,1.0], [(2,0),(0,1)], [1.0,1.0], [(0,0),(1,2),(2,1)], [1.0,1.0,1.0]))
        terms.append(([(1,0),(2,2)], [1.0,1.0], [(0,2),(1,1)], [1.0,1.0], [(1,0),(2,2),(0,1)], [1.0,1.0,1.0]))
        terms.append(([(1,1),(2,0)], [1.0,1.0], [(1,2),(2,0)], [1.0,1.0], [(1,1),(2,0),(0,2)], [1.0,1.0,1.0]))
        terms.append(([(1,2),(2,1)], [1.0,1.0], [(2,1),(0,2)], [1.0,1.0], [(1,2),(2,2),(0,0)], [1.0,1.0,1.0]))
        terms.append(([(0,0),(2,2)], [1.0,1.0], [(0,0),(2,2)], [1.0,1.0], [(0,0),(1,1),(2,2)], [1.0,1.0,1.0]))
        terms.append(([(0,1),(2,0)], [1.0,1.0], [(1,0),(2,1)], [1.0,1.0], [(0,1),(1,0),(2,2)], [1.0,1.0,1.0]))
        terms.append(([(0,2),(2,1)], [1.0,1.0], [(2,0),(0,2)], [1.0,1.0], [(0,2),(1,2),(2,0)], [1.0,1.0,1.0]))
        
        # 19-23: Triples (Winograd)
        # These are tricky. I will skip the complex Winograd terms if I'm not sure.
        # Let's just use the 18 terms above? No, that's not a valid decomposition.
        
        # Fallback: Use Schoolbook 27 for 3x3 blocks.
        # This gives rank 189.
        return None

    # Since BCLR is risky, I will stick to the current 189 solution.
    # But I can try to optimize the Strassen constants? No.
    
    # I will return the current solution.
    bs = 2
    nb = n // bs
    
    strassen_defs = [
        {'A': {0: 1.0, 3: 1.0}, 'B': {0: 1.0, 3: 1.0}},
        {'A': {2: 1.0, 3: 1.0}, 'B': {0: 1.0}},
        {'A': {0: 1.0}, 'B': {1: 1.0, 3: -1.0}},
        {'A': {3: 1.0}, 'B': {2: 1.0, 0: -1.0}},
        {'A': {0: 1.0, 1: 1.0}, 'B': {3: 1.0}},
        {'A': {2: 1.0, 0: -1.0}, 'B': {0: 1.0, 1: 1.0}},
        {'A': {1: 1.0, 3: -1.0}, 'B': {2: 1.0, 3: 1.0}},
    ]
    
    recon = [
        {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
        {2: 1.0, 4: 1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 1: -1.0, 2: 1.0, 5: 1.0},
    ]
    
    rank = nb * nb * nb * 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    for blk, coeff in s_def['A'].items():
                        ar = blk // 2
                        ac = blk % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U[i_glob * m + k_glob, r_idx] += coeff

                    for blk, coeff in s_def['B'].items():
                        br = blk // 2
                        bc = blk % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V[k_glob * p + j_glob, r_idx] += coeff

                    for c_blk in range(4):
                        gamma = recon[c_blk].get(s, 0.0)
                        if gamma != 0.0:
                            cr = c_blk // 2
                            cc = c_blk % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W[j_glob * n + i_glob, r_idx] += gamma

                    r_idx += 1

    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank_t = n * m * p
    U_t = np.zeros((n * m, rank_t), dtype=float)
    V_t = np.zeros((m * p, rank_t), dtype=float)
    W_t = np.zeros((p * n, rank_t), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_t[i * m + k, idx] = 1.0
                V_t[k * p + j, idx] = 1.0
                W_t[j * n + i, idx] = 1.0
                idx += 1
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
