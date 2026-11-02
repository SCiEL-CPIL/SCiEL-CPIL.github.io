import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # The 16 columns of B are grouped into 8 blocks of 2 columns.
    # Each block is a 2x2 matrix multiplication.
    # Rank of 2x2x2 is 7.
    # Total rank = 8 * 7 = 56.
    
    num_blocks = p // 2
    R = num_blocks * 7

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # Indices in A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices in B (local): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices in C (local): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_coeffs = [
        {0: 1, 3: 1}, # a+d
        {2: 1, 3: 1}, # c+d
        {0: 1},       # a
        {3: 1},       # d
        {0: 1, 1: 1}, # a+b
        {2: 1, 0: -1},# c-a
        {1: 1, 3: -1} # b-d
    ]
    
    V_coeffs = [
        {0: 1, 3: 1}, # e+h
        {0: 1},       # e
        {1: 1, 3: -1},# f-h
        {2: 1, 0: -1},# g-e
        {3: 1},       # h
        {0: 1, 1: 1}, # e+f
        {2: 1, 3: 1}  # g+h
    ]
    
    W_contribs = [
        {0: 1, 3: 1}, # m1+m4-m5+m7 -> C00
        {2: 1, 3: -1},# m2+m4 -> C10 (Note: W index mapping)
        {1: 1, 3: 1}, # m3+m5 -> C01
        {0: 1, 2: 1}, # m1-m2+m3+m6 -> C11
        {0: -1, 1: 1},# m1+m4-m5+m7 ... wait
        # Let's re-verify W mapping
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # My previous W_contribs in the prompt:
        # 0: {0: 1, 3: 1} -> m1, m4? No.
        # The prompt's W_contribs were:
        # 0: {0: 1, 3: 1}
        # 1: {2: 1, 3: -1}
        # 2: {1: 1, 3: 1}
        # 3: {0: 1, 2: 1}
        # 4: {0: -1, 1: 1}
        # 5: {3: 1}
        # 6: {0: 1}
        
        # Let's map s=0..6 to m1..m7
        # s=0: m1. Coeffs: C00(+1), C11(+1)
        # s=1: m2. Coeffs: C10(+1), C11(-1)
        # s=2: m3. Coeffs: C01(+1), C11(+1)
        # s=3: m4. Coeffs: C00(+1), C10(+1)
        # s=4: m5. Coeffs: C00(-1), C01(+1)
        # s=5: m6. Coeffs: C11(+1)
        # s=6: m7. Coeffs: C00(+1)
        
        # Local indices:
        # 0: C00
        # 1: C01
        # 2: C10
        # 3: C11
        
        # s=0 (m1): C00(+1), C11(+1) -> {0: 1, 3: 1}
        # s=1 (m2): C10(+1), C11(-1) -> {2: 1, 3: -1}
        # s=2 (m3): C01(+1), C11(+1) -> {1: 1, 3: 1}
        # s=3 (m4): C00(+1), C10(+1) -> {0: 1, 2: 1}
        # s=4 (m5): C00(-1), C01(+1) -> {0: -1, 1: 1}
        # s=5 (m6): C11(+1) -> {3: 1}
        # s=6 (m7): C00(+1) -> {0: 1}
    ]
    
    W_contribs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]

    for q in range(num_blocks):
        # B indices for block q
        # B is 2x16. Flattened: B[0,0], B[0,1]... B[0,15], B[1,0]...
        # V shape is (m*p, R) = (32, R)
        # Row index j*p + k
        # For block q (columns 2q, 2q+1):
        # B[0, 2q] is at index 0*p + 2q = 2q
        # B[0, 2q+1] is at index 2q+1
        # B[1, 2q] is at index 1*p + 2q = 16 + 2q
        # B[1, 2q+1] is at index 16 + 2q + 1
        
        b00_idx = 2 * q
        b01_idx = 2 * q + 1
        b10_idx = 16 + 2 * q
        b11_idx = 16 + 2 * q + 1
        
        # W indices for block q
        # W shape is (p*n, R) = (32, R)
        # Row index k*n + i
        # For block q (columns 2q, 2q+1):
        # C[0, 2q] is at index 2q*n + 0 = 4q
        # C[0, 2q+1] is at index (2q+1)*n + 0 = 4q+2
        # C[1, 2q] is at index 2q*n + 1 = 4q+1
        # C[1, 2q+1] is at index (2q+1)*n + 1 = 4q+3
        
        c00_idx = 4 * q
        c01_idx = 4 * q + 2
        c10_idx = 4 * q + 1
        c11_idx = 4 * q + 3
        
        for s in range(7):
            r = q * 7 + s
            
            # U
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
                
            # V
            for local_idx, coeff in V_coeffs[s].items():
                if local_idx == 0:
                    V[b00_idx, r] = coeff
                elif local_idx == 1:
                    V[b01_idx, r] = coeff
                elif local_idx == 2:
                    V[b10_idx, r] = coeff
                else:
                    V[b11_idx, r] = coeff
                    
            # W
            for local_out_idx, coeff in W_contribs[s].items():
                if local_out_idx == 0:
                    W[c00_idx, r] = coeff
                elif local_out_idx == 1:
                    W[c01_idx, r] = coeff
                elif local_out_idx == 2:
                    W[c10_idx, r] = coeff
                else:
                    W[c11_idx, r] = coeff

    # Verification
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        C_test_flat = C_test.T.flatten()

        if not np.allclose(C_decomp, C_test_flat):
            # Fallback to trivial decomposition
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)

            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
