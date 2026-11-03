import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0
    
    # View 6x6 as 3x3 blocks of 2x2 matrices
    # Block size: 2, number of blocks: 3
    bs = 2
    nb = 3  # 3x3 blocks
    
    # Rank: 27 block products (i,k,j in 0..2) * 7 Strassen terms
    rank = 27 * 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 definitions
    # A_s = sum coeff * A_block, B_s = sum coeff * B_block
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11 (local 2x2 blocks within the 2x2 sub-matrix)
    # But wait - for 2x2 matrix multiplication, the "blocks" are individual elements
    # Actually for 2x2 * 2x2, Strassen works on the 4 elements directly
    
    # For 2x2 matrix multiplication C = A*B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # Strassen:
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01 = M2 + M4
        [(2, 1.0), (4, 1.0)],
        # C10 = M1 + M3
        [(1, 1.0), (3, 1.0)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for i_blk in range(nb):
        for k_blk in range(nb):
            for j_blk in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    
                    # For this Strassen term, we need to map:
                    # A element: local (a_row, a_col) within block (i_blk, k_blk)
                    # B element: local (b_row, b_col) within block (k_blk, j_blk)
                    # C element: local (c_row, c_col) within block (i_blk, j_blk)
                    
                    # U: A[i_global, k_global] where i_global = i_blk*bs + a_row, k_global = k_blk*bs + a_col
                    for a_row in range(bs):
                        for a_col in range(bs):
                            # A_s[a_row, a_col] = sum coeff * A_local[a_row, a_col]
                            # A_local is the 2x2 block, elements indexed 0,1,2,3 = (0,0),(0,1),(1,0),(1,1)
                            local_idx = a_row * bs + a_col  # 0,1,2,3
                            for block_idx, coeff_a in s_def['A']:
                                # block_idx refers to which element of the 2x2 A matrix
                                # But wait - for 2x2, the "blocks" in Strassen are the elements themselves
                                # block_idx 0 = element (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
                                # So block_idx == local_idx
                                if block_idx == local_idx:
                                    i_glob = i_blk * bs + a_row
                                    k_glob = k_blk * bs + a_col
                                    u_row = i_glob * m + k_glob
                                    U[u_row, r_idx] += coeff_a
                    
                    # V: B[k_global, j_global] where k_global = k_blk*bs + b_row, j_global = j_blk*bs + b_col
                    for b_row in range(bs):
                        for b_col in range(bs):
                            local_idx = b_row * bs + b_col
                            for block_idx, coeff_b in s_def['B']:
                                if block_idx == local_idx:
                                    k_glob = k_blk * bs + b_row
                                    j_glob = j_blk * bs + b_col
                                    v_row = k_glob * p + j_glob
                                    V[v_row, r_idx] += coeff_b
                    
                    # W: C[i_global, j_global] where i_global = i_blk*bs + c_row, j_global = j_blk*bs + c_col
                    for c_row in range(bs):
                        for c_col in range(bs):
                            local_idx = c_row * bs + c_col
                            for c_idx, gamma in recon_coeffs[local_idx]:
                                if c_idx == s:
                                    i_glob = i_blk * bs + c_row
                                    j_glob = j_blk * bs + c_col
                                    w_row = j_glob * n + i_glob
                                    W[w_row, r_idx] += gamma
                    
                    r_idx += 1
    
    assert r_idx == rank, f"r_idx={r_idx}, rank={rank}"
    
    # Fast verification using tensor contraction
    # T[u,v,w] = 1 if u=i*m+k, v=k*p+j, w=j*n+i
    # Reconstructed: sum_r U[u,r]*V[v,r]*W[w,r]
    
    # Build target tensor sparsely
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T_target[i * m + k, k * p + j, j * n + i] = 1.0
    
    # Verify with deadline check
    if time.time() > deadline - 10.0:
        # Not enough time for full verification, trust the construction
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to naive
        rank_naive = n * m * p
        U_std = np.zeros((n * m, rank_naive), dtype=float)
        V_std = np.zeros((m * p, rank_naive), dtype=float)
        W_std = np.zeros((p * n, rank_naive), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_std[i * m + k, idx] = 1.0
                    V_std[k * p + j, idx] = 1.0
                    W_std[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
