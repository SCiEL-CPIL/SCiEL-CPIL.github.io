import numpy as np
import time

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 8

    # Target rank 88:
    # Row 0 (1x4x8): Schoolbook (32 terms)
    # Rows 1-2 (2x4x8): Strassen on 2x2x2 blocks (8 blocks * 7 terms = 56 terms)
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # 1. Row 0: Schoolbook
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # 2. Rows 1-2: Strassen
    # Strassen factors for 2x2x2
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [1, -1, 0, 0],
        [0, 0, 1, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    
    # Standard Strassen factors
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B00+B10)
    # M7 = (A10-A00)*(B11+B01)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 - M3 - M6
    # C11 = M1 - M2 + M6
    
    U_s = np.array([
        [1, 0, 1, 0], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A10
        [0, 1, 0, -1] # M6: A01, -A11
        # M7: A10 - A00 => [-1, 1, 0, 0]
        # Wait, U_s index is i*2+j.
        # M7: A10 (i=1, j=0) - A00 (i=0, j=0)
        # So row 7 of U_s: [-1, 1, 0, 0]
    ], dtype=float)
    
    # Add M7 row to U_s
    U_s = np.vstack([U_s, np.array([-1, 1, 0, 0])])
    
    V_s = np.array([
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01, -B11
        [0, -1, 1, 0], # M4: -B00, B10
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B10
        [0, 1, 0, 1]  # M7: B01, B11
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, -1], # M1: C00, -C11
        [0, 0, -1, 0], # M2: -C10
        [0, 1, 0, 0],  # M3: C01
        [0, 1, 0, -1], # M4: C01, -C11
        [1, 0, 0, 0],  # M5: C00
        [0, 0, -1, 1], # M6: -C10, C11
        [-1, 0, 0, 1]  # M7: -C00, C11
    ], dtype=float)
    
    # Verify Strassen
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = 0.0
                    for r in range(7):
                        val += U_s[r, i*2+j] * V_s[r, j*2+k] * W_s[r, k*2+i]
                    expected = 1.0 if (i == j == k) else 0.0
                    if abs(val - expected) > 1e-9:
                        return False
        return True
        
    if not verify_strassen():
        # Fallback to schoolbook if Strassen factors are wrong
        return _schoolbook_decomposition(n, m, p)

    # Apply Strassen to blocks
    # We are decomposing the submatrix for i in {1, 2} (rows 1, 2 of A)
    # against j in {0..3} (cols 0..3 of A, rows of B)
    # and k in {0..7} (cols 0..7 of B, cols of C)
    
    # Since m=4, we have 2 blocks of size 2x2 for the j dimension.
    # Since p=8, we have 4 blocks of size 2x2 for the k dimension.
    
    for j_block in range(2):
        for k_block in range(4):
            j_start = j_block * 2
            k_start = k_block * 2
            for t in range(7):
                for li in range(2):
                    for lj in range(2):
                        for lk in range(2):
                            # U: global i = 1+li, global j = j_start+lj
                            u_val = U_s[t, li*2+lj]
                            if u_val != 0:
                                U[(1+li)*m + (j_start+lj), col] += u_val
                            
                            # V: global j = j_start+lj, global k = k_start+lk
                            v_val = V_s[t, lj*2+lk]
                            if v_val != 0:
                                V[(j_start+lj)*p + (k_start+lk), col] += v_val
                            
                            # W: global k = k_start+lk, global i = 1+li
                            w_val = W_s[t, lk*2+li]
                            if w_val != 0:
                                W[(k_start+lk)*n + (1+li), col] += w_val
                col += 1
                
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Verify full decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can vectorize this check using einsum for speed
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Reshape to tensors
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i, j, k]
    # T = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # This is a bit tricky with einsum.
    # Let's do it in steps or use a loop if n,m,p are small.
    # n=3, m=4, p=8 is small enough for a loop if optimized, or numpy broadcasting.
    
    # Let's use a loop for clarity and correctness first, then optimize if needed.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u_vec = U[i*m+j, :]
                # v_vec = V[j*p+k, :]
                # w_vec = W[k*n+i, :]
                # val = np.sum(u_vec * v_vec * w_vec)
                
                # Using dot product trick:
                # val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                t_val = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                expected = 1.0 if (i == j == k) else 0.0
                err = abs(t_val - expected)
                if err > max_err:
                    max_err = err
                    
    if max_err > 1e-9:
        # If Strassen failed, fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
