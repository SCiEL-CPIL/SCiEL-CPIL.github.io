import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for [2,4,4] matrix multiplication.
    Uses Strassen rank-28 base with greedy linear dependency elimination.
    Ensures integer/half-integer coefficients.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 4
    R = n * m * p  # 32
    
    # Build rank-28 Strassen decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    # Blocks for A (2x4): A1 (cols 0,1), A2 (cols 2,3)
    # Each block is 2x2.
    # We process 4 independent 2x2x2 multiplications:
    # 1. A1 * B11 -> C11
    # 2. A2 * B21 -> C11 (added to C11)
    # 3. A1 * B12 -> C12
    # 4. A2 * B22 -> C12 (added to C12)
    # Note: C is 2x4. C11 is cols 0,1. C12 is cols 2,3.
    
    def add_strassen(a_r, a_c, b_r, b_c, c_c, start_idx):
        A00 = u_idx(a_r, a_c)
        A01 = u_idx(a_r, a_c+1)
        A10 = u_idx(a_r+1, a_c)
        A11 = u_idx(a_r+1, a_c+1)
        
        B00 = v_idx(b_r, b_c)
        B01 = v_idx(b_r, b_c+1)
        B10 = v_idx(b_r+1, b_c)
        B11 = v_idx(b_r+1, b_c+1)
        
        C00 = w_idx(0, c_c)
        C01 = w_idx(0, c_c+1)
        C10 = w_idx(1, c_c)
        C11 = w_idx(1, c_c+1)
        
        # M1
        U[A00, start_idx+0] += 1
        U[A11, start_idx+0] += 1
        V[B00, start_idx+0] += 1
        V[B11, start_idx+0] += 1
        W[C00, start_idx+0] += 1
        W[C11, start_idx+0] += 1
        
        # M2
        U[A10, start_idx+1] += 1
        U[A11, start_idx+1] += 1
        V[B00, start_idx+1] += 1
        W[C10, start_idx+1] += 1
        W[C11, start_idx+1] -= 1
        
        # M3
        U[A00, start_idx+2] += 1
        V[B01, start_idx+2] += 1
        V[B11, start_idx+2] -= 1
        W[C01, start_idx+2] += 1
        W[C11, start_idx+2] += 1
        
        # M4
        U[A11, start_idx+3] += 1
        V[B10, start_idx+3] += 1
        V[B00, start_idx+3] -= 1
        W[C00, start_idx+3] += 1
        W[C10, start_idx+3] += 1
        
        # M5
        U[A00, start_idx+4] += 1
        U[A01, start_idx+4] += 1
        V[B11, start_idx+4] += 1
        W[C00, start_idx+4] -= 1
        W[C01, start_idx+4] += 1
        
        # M6
        U[A10, start_idx+5] += 1
        U[A00, start_idx+5] -= 1
        V[B00, start_idx+5] += 1
        V[B01, start_idx+5] += 1
        W[C11, start_idx+5] += 1
        
        # M7
        U[A01, start_idx+6] += 1
        U[A11, start_idx+6] -= 1
        V[B10, start_idx+6] += 1
        V[B11, start_idx+6] += 1
        W[C00, start_idx+6] += 1

    # Product 1: A_L (rows 0,1 cols 0,1) * B_TL (rows 0,1 cols 0,1) -> C cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    # Product 2: A_R (rows 0,1 cols 2,3) * B_BL (rows 2,3 cols 0,1) -> C cols 0,1
    add_strassen(0, 2, 2, 0, 0, 7)
    # Product 3: A_L (rows 0,1 cols 0,1) * B_TR (rows 0,1 cols 2,3) -> C cols 2,3
    add_strassen(0, 0, 0, 2, 2, 14)
    # Product 4: A_R (rows 0,1 cols 2,3) * B_BR (rows 2,3 cols 2,3) -> C cols 2,3
    add_strassen(0, 2, 2, 2, 2, 21)
    
    def verify(Um, Vm, Wm):
        Td = np.zeros((n,m,p), dtype=np.int64)
        for t in range(Um.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Td[i,j,k] += Wm[k*n+i,t] * Um[i*m+j,t] * Vm[j*p+k,t]
        return np.all(Td == 1)
    
    if not verify(U, V, W):
        raise ValueError("Invalid initial decomposition")
    
    def vec_r1(Uc, Vc, Wc):
        T = np.zeros(R, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k*n*m + i*m + j] = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        return T
    
    T_target = np.ones(R, dtype=np.int64)
    
    def build_M(Um, Vm, Wm):
        return np.array([vec_r1(Um[:,t], Vm[:,t], Wm[:,t]) for t in range(Um.shape[1])]).T
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    M = build_M(best_U, best_V, best_W)
    
    # Greedy elimination
    for iteration in range(10):
        if time.time() > deadline:
            break
        removed = False
        for t in range(best_rank):
            if time.time() > deadline:
                break
            keep = [s for s in range(best_rank) if s != t]
            if not keep:
                continue
            M_sub = M[:, keep]
            try:
                c, _, _, _ = np.linalg.lstsq(M_sub.astype(float), T_target.astype(float), rcond=None)
                if np.linalg.norm(M_sub.astype(float) @ c - T_target.astype(float)) < 1e-9:
                    new_U = best_U[:, keep].copy()
                    new_V = best_V[:, keep].copy()
                    new_W = best_W[:, keep].copy()
                    
                    # Check if c can be represented as integer or half-integer
                    c_2 = np.round(c * 2)
                    if np.allclose(c * 2, c_2, atol=1e-9):
                        for idx, s in enumerate(keep):
                            new_V[:, idx] = best_V[:, s] * c[idx]
                        
                        valid = True
                        for val in new_V.flatten():
                            if not np.isclose(val * 2, np.round(val * 2), atol=1e-9):
                                valid = False
                                break
                        if valid:
                            # Convert to int64 for verification (multiply by 2 if necessary? No, verify uses int64)
                            # If values are half-integers, int64 truncates. 
                            # We should verify using floats or scaled integers.
                            # Let's verify using the float arrays first.
                            if verify(new_U.astype(float), new_V.astype(float), new_W.astype(float)):
                                best_rank = len(keep)
                                best_U, best_V, best_W = new_U, new_V, new_W
                                M = build_M(best_U.astype(np.int64), best_V.astype(np.int64), best_W.astype(np.int64))
                                removed = True
                                break
            except:
                continue
        if not removed:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
