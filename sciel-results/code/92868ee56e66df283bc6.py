import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    n, m, p = DIMENSIONS

    # Strategy:
    # 1. Decompose 4x4x5 into 4x4x4 and 4x4x1.
    # 2. Use Strassen recursion for 4x4x4 (Rank 49).
    # 3. Use Schoolbook for 4x4x1 (Rank 16).
    # Total Rank: 65.
    
    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    
    # Strassen 2x2x2 Rank 7 Decomposition
    # T_222 = sum_t u_t (x) v_t (x) w_t
    # Indices for 2x2 matrices are flattened: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # The 7 terms of Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> W: C00, C11
    # M2 = (A10+A11)B00 -> W: C10
    # M3 = A00(B10-B11) -> W: C01, C10
    # M4 = A11(B01-B00) -> W: C00, C01
    # M5 = (A00+A01)B11 -> W: C00, C11
    # M6 = (A10-A00)(B00+B01) -> W: C10
    # M7 = (A01-A11)(B10+B11) -> W: C01, C11
    
    # u_t, v_t, w_t are vectors in R^4.
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = np.array([1, 0, 0, 1]) # A00, A11
    v1 = np.array([1, 0, 0, 1]) # B00, B11
    w1 = np.array([1, 0, 0, 1]) # C00, C11
    
    # Term 2: (A10+A11)B00
    u2 = np.array([0, 0, 1, 1]) # A10, A11
    v2 = np.array([1, 0, 0, 0]) # B00
    w2 = np.array([0, 0, 1, 0]) # C10
    
    # Term 3: A00(B10-B11)
    u3 = np.array([1, 0, 0, 0]) # A00
    v3 = np.array([0, 0, 1, -1]) # B10, -B11
    w3 = np.array([0, 1, 1, 0]) # C01, C10
    
    # Term 4: A11(B01-B00)
    u4 = np.array([0, 0, 0, 1]) # A11
    v4 = np.array([-1, 1, 0, 0]) # -B00, B01
    w4 = np.array([1, 1, 0, 0]) # C00, C01
    
    # Term 5: (A00+A01)B11
    u5 = np.array([1, 1, 0, 0]) # A00, A01
    v5 = np.array([0, 0, 0, 1]) # B11
    w5 = np.array([1, 0, 0, 1]) # C00, C11
    
    # Term 6: (A10-A00)(B00+B01)
    u6 = np.array([-1, 0, 1, 0]) # -A00, A10
    v6 = np.array([1, 1, 0, 0]) # B00, B01
    w6 = np.array([0, 0, 1, 0]) # C10
    
    # Term 7: (A01-A11)(B10+B11)
    u7 = np.array([0, 1, 0, -1]) # A01, -A11
    v7 = np.array([0, 0, 1, 1]) # B10, B11
    w7 = np.array([0, 1, 0, 1]) # C01, C11
    
    inner_u = [u1, u2, u3, u4, u5, u6, u7]
    inner_v = [v1, v2, v3, v4, v5, v6, v7]
    inner_w = [w1, w2, w3, w4, w5, w6, w7]
    
    # Block Level Strassen Coefficients for 4x4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block terms s=1..7
    # alpha_s: coefficients for A blocks (2x2 matrix of coeffs)
    # beta_s: coefficients for B blocks
    # gamma_s: coefficients for C blocks
    
    # M1 = (A00+A11)(B00+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    
    # alpha_s (2x2 block coeffs for A)
    # Indexing: [00, 01, 10, 11]
    alpha = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [1, 0, -1, 0],# M6: A00 - A10 (Wait, M6 is (A10-A00), so -A00 + A10) -> [-1, 0, 1, 0]
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    # Correction for M6: (A10 - A00) -> A00 coeff -1, A10 coeff 1.
    alpha[5] = [-1, 0, 1, 0]
    
    # beta_s (2x2 block coeffs for B)
    beta = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10 - B11
        [-1, 1, 0, 0],# M4: -B00 + B01
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # gamma_s (2x2 block coeffs for C)
    # C00, C01, C10, C11
    gamma = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 1, 0], # M3: C01 + C10
        [1, 1, 0, 0], # M4: C00 + C01
        [1, 0, 0, 1], # M5: C00 + C11
        [0, 0, 1, 0], # M6: C10
        [0, 1, 0, 1]  # M7: C01 + C11
    ]
    
    # Now construct the 49 terms for 4x4x4
    # Global U, V, W for the 4x4x4 part
    # Shapes: U (16, 49), V (16, 49), W (16, 49)
    # Note: For 4x4x4, W index is k*4+i.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Construct U_col
            # U_col corresponds to A (4x4)
            # A is partitioned into 2x2 blocks of 2x2.
            # Block (a,b) of A is at rows a*2..a*2+1, cols b*2..b*2+1.
            # alpha[s] gives coeffs for blocks 00, 01, 10, 11.
            # inner_u[t] gives coeffs for elements 00, 01, 10, 11 within a 2x2 block.
            
            # Flatten A indices: i*4+j
            for a in range(2):
                for b in range(2):
                    # Block index in alpha: a*2+b
                    coeff_alpha = alpha[s][a*2+b]
                    if coeff_alpha == 0:
                        continue
                    for r in range(2):
                        for c in range(2):
                            # Element index in inner_u: r*2+c
                            coeff_u = inner_u[t][r*2+c]
                            if coeff_u == 0:
                                continue
                            # Global A index: (a*2+r)*4 + (b*2+c)
                            idx = (a*2+r)*4 + (b*2+c)
                            U_444[idx, col] += coeff_alpha * coeff_u
            
            # Construct V_col
            # V corresponds to B (4x4)
            # B partitioned into 2x2 blocks.
            # beta[s] gives coeffs for blocks.
            # inner_v[t] gives coeffs for elements.
            for c in range(2): # Block row
                for d in range(2): # Block col
                    coeff_beta = beta[s][c*2+d]
                    if coeff_beta == 0:
                        continue
                    for r in range(2):
                        for k in range(2):
                            coeff_v = inner_v[t][r*2+k]
                            if coeff_v == 0:
                                continue
                            # Global B index: (c*2+r)*4 + (d*2+k)
                            idx = (c*2+r)*4 + (d*2+k)
                            V_444[idx, col] += coeff_beta * coeff_v
            
            # Construct W_col
            # W corresponds to C (4x4)
            # C partitioned into 2x2 blocks.
            # gamma[s] gives coeffs for blocks.
            # inner_w[t] gives coeffs for elements.
            # Note: W index convention is k*n+i (col*4+row)
            # C block (i,j) is at rows i*2.., cols j*2..
            # gamma[s] index: i*2+j
            for i in range(2):
                for j in range(2):
                    coeff_gamma = gamma[s][i*2+j]
                    if coeff_gamma == 0:
                        continue
                    for r in range(2):
                        for c in range(2):
                            coeff_w = inner_w[t][r*2+c]
                            if coeff_w == 0:
                                continue
                            # Global C element: row (i*2+r), col (j*2+c)
                            # W index: col*4 + row = (j*2+c)*4 + (i*2+r)
                            idx = (j*2+c)*4 + (i*2+r)
                            W_444[idx, col] += coeff_gamma * coeff_w
            
            col += 1
    
    # --- Part 2: 4x4x1 Schoolbook Decomposition (Rank 16) ---
    # A (4x4) * B (4x1) = C (4x1)
    # U (16, 16), V (4, 16), W (4, 16)
    # Wait, dimensions:
    # U: n*m = 4*4 = 16
    # V: m*p = 4*1 = 4
    # W: p*n = 1*4 = 4
    
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            # C[i,0] += A[i,j] * B[j,0]
            U_441[i*4+j, col] = 1
            V_441[j, col] = 1
            W_441[0*4+i, col] = 1 # k=0, so 0*4+i = i
            col += 1
    
    # --- Combine ---
    # Total Rank = 49 + 16 = 65
    total_rank = 65
    
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank)) # m*p = 4*5 = 20
    W = np.zeros((20, total_rank)) # p*n = 5*4 = 20
    
    # Fill 4x4x4 part (cols 0..48)
    U[:, 0:49] = U_444
    V[:, 0:49] = V_444
    W[:, 0:49] = W_444
    
    # Fill 4x4x1 part (cols 49..64)
    # A is 4x4, B is 4x1 (part of 4x5).
    # B indices 0..3 correspond to the first 4 columns of B?
    # We split B (4x5) into B1 (4x4) and B2 (4x1).
    # So V for 4x4x1 part uses B indices 0..3.
    V[0:4, 49:65] = V_441
    W[0:4, 49:65] = W_441
    U[:, 49:65] = U_441
    
    # Verify
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 5)
    C_expected = A @ B
    C_computed = np.zeros((4, 5))
    for i in range(4):
        for k in range(5):
            for r in range(total_rank):
                u_val = np.dot(U[:, r], A.flatten())
                v_val = np.dot(V[:, r], B.flatten())
                C_computed[i, k] += u_val * v_val * W[k*4 + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-5):
        # Fallback to schoolbook if verification fails (should not happen)
        U = np.zeros((16, 80))
        V = np.zeros((20, 80))
        W = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_dot_a = np.dot(U[:, r], A.flatten())
                    v_dot_b = np.dot(V[:, r], B.flatten())
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
    
    return True

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
