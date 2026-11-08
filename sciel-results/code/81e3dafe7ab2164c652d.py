import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
    # 1 Schoolbook block (2x2x1) for k=6 -> 4
    # 1 Schoolbook block (2x1x7) for j=2 -> 14
    rank = 21 + 4 + 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    # Helper for Strassen block
    # local A indices: 00, 01, 10, 11 -> global i*3+j
    # local B indices: 00, 01, 10, 11 -> global j*7+k
    # local C indices: 00, 01, 10, 11 -> global k*2+i
    def add_strassen_block(k_start):
        # k_start is 0, 2, 4
        # A indices:
        # A00: 0*3+0 = 0
        # A01: 0*3+1 = 1
        # A10: 1*3+0 = 3
        # A11: 1*3+1 = 4
        
        # B indices:
        # B00: 0*7 + k_start
        # B01: 0*7 + k_start + 1
        # B10: 1*7 + k_start
        # B11: 1*7 + k_start + 1
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices:
        # C00: k_start*2 + 0
        # C01: (k_start+1)*2 + 0
        # C10: k_start*2 + 1
        # C11: (k_start+1)*2 + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    # Add 3 Strassen blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block 4: k=6, 2x2x1 Schoolbook
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part 2: j=2
    # A02 B2k C0k
    # A12 B2k C1k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    # Verify decomposition correctness before returning
    _elapsed = time.time() - _t0
    if _elapsed < 140:  # safety margin before 150s kill
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if (i,j,k) is a valid triple, 0 otherwise
            # The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
            # Actually: T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j] B[j,k]
            # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            n_chk, m_chk, p_chk = n, m, p
            err = 0.0
            for i in range(n_chk):
                for j in range(m_chk):
                    for k in range(p_chk):
                        s = 0.0
                        for r in range(rank):
                            s += Ua[i*m_chk+j, r] * Va[j*p_chk+k, r] * Wa[k*n_chk+i, r]
                        if abs(s - 1.0) > 1e-9:
                            err += abs(s - 1.0)
            if err > 1e-6:
                # Fallback: return a known-valid schoolbook decomposition
                rank_sb = n * m * p
                U_sb = [[0] * rank_sb for _ in range(n * m)]
                V_sb = [[0] * rank_sb for _ in range(m * p)]
                W_sb = [[0] * rank_sb for _ in range(p * n)]
                col_sb = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_sb[i*m+j][col_sb] = 1
                            V_sb[j*p+k][col_sb] = 1
                            W_sb[k*n+i][col_sb] = 1
                            col_sb += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
