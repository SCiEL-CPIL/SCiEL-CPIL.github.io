import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    # Use rank 14: two independent rank-7 algorithms for each row of A
    # Each row i: c_i = a_i * B where a_i is 1x4, B is 4x4
    # We need a bilinear algorithm for 1x4 * 4x4 = 1x4 with rank 7
    
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each row i (i=0,1), we have 7 terms
    # Term r within row i: r_global = i*7 + r_local (r_local in 0..6)
    
    # Rank-7 algorithm for 1x4 * 4x4:
    # c_k = sum_j a_j * b_jk
    # We want: c_k = sum_{r=0}^{6} alpha_{kr} * (sum_j beta_{jr} * a_j) * (sum_l gamma_{lr} * b_{j'l'...})
    # Actually: c_k = sum_r W_{kr} * (U_r . a) * (V_r . b)
    # where U_r is 1x4, V_r is 4x4 flattened to 16, W_r is 4x1
    
    # Let's use a structured approach:
    # Split the 4 columns into two groups of 2: {0,1} and {2,3}
    # For each group, use a 1x2 * 2x2 multiplication with rank 3 (Strassen-like)
    # But 1x2 * 2x2 = 1x2, schoolbook rank is 4, can we do better?
    # 1x2 * 2x2: c_0 = a_0*b_00 + a_1*b_10, c_1 = a_0*b_01 + a_1*b_11
    # This is rank 4 (each output needs 2 products, no sharing)
    # Actually we can share: let d0 = a_0*b_00, d1 = a_0*b_01, d2 = a_1*b_10, d3 = a_1*b_11
    # That's rank 4, no improvement for 1x2*2x2.
    
    # Alternative: use rank 7 by combining both halves with cross terms.
    # Let me try a different approach: use the known fact that 1x4*4x4 has tensor rank at most 8 (schoolbook)
    # Can we do 7? Let's try to construct it.
    
    # c_k = sum_j a_j b_jk for k=0,1,2,3
    # 4 outputs, each is a dot product of a (4-dim) with column k of B.
    # Total: 4*4 = 16 scalar products in schoolbook.
    # We need to express as sum of 7 rank-1 terms.
    
    # Let me try: define 7 linear combinations of a, 7 linear combinations of B (flattened), and 7 linear combinations for output.
    # This is complex. Let me instead use a practical approach:
    # Use rank 8 (schoolbook for each row) -> total rank 16, then try to optimize.
    # But let's first get a valid rank 14 solution.
    
    # Actually, let me use a simpler valid approach: rank 8 per row = 16 total.
    # Then try to reduce to 14.
    
    # For now, let's construct rank 16 (8 per row) as a fallback, then try rank 14.
    
    # Rank 8 per row (schoolbook for 1x4 * 4x4):
    # c_k = sum_j a_j b_jk
    # Term (j,k): U_{j, r} = a_j coefficient, V_{jk, r} = b_jk coefficient, W_{k, r} = c_k coefficient
    # r = j*4 + k for the 16 terms... wait that's rank 16 per row, total 32.
    # No: for 1x4 * 4x4, schoolbook is rank 16 (4*4 products).
    # But we only need 4 outputs, each requiring 4 products = 16 total, but they don't share.
    # So rank 16 per row, total 32. That's the same as before.
    
    # Wait, I'm confusing myself. Let me restart.
    # n=2, m=4, p=4.
    # A is 2x4, B is 4x4, C is 2x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Schoolbook: 2*4*4 = 32 multiplications.
    
    # For rank 14, we need a clever decomposition.
    # Let's try: treat the two rows of A separately.
    # Row 0: C[0,k] = sum_j A[0,j] B[j,k] for k=0,1,2,3
    # Row 1: C[1,k] = sum_j A[1,j] B[j,k] for k=0,1,2,3
    
    # Each row is a 1x4 * 4x4 multiplication.
    # The tensor for 1x4 * 4x4 has rank at most 16 (schoolbook).
    # Can we do better? The border rank might be lower.
    
    # Let me try a practical approach: use rank 7 per row if possible, else 8.
    # I'll construct a valid rank 14 decomposition using a heuristic search.
    
    # Actually, let me just use the known Strassen-based approach for 2x4x4.
    # Split A into [A0 | A1] where A0, A1 are 2x2.
    # Split B into [[B00, B01], [B10, B11]] where each is 2x2.
    # Then C = [A0*B00 + A1*B10 | A0*B01 + A1*B11]
    # Each 2x2*2x2 can be done in rank 7 (Strassen).
    # C_left = A0*B00 + A1*B10: this is a sum of two 2x2*2x2 products.
    # We can compute A0*B00 and A1*B10 separately using Strassen, but that's 7+7=14 per half.
    # Total: 28 for both halves. That's what the inspiration mentioned.
    
    # For rank 14, we need more sharing.
    # Let me try: compute M1 = A0*B00, M2 = A1*B10, M3 = A0*B01, M4 = A1*B11.
    # C_left = M1 + M2, C_right = M3 + M4.
    # If we compute each Mi using Strassen (rank 7 each), total rank 28.
    # To get rank 14, we'd need to share computations across M1,M2,M3,M4.
    
    # This is getting complex. Let me implement a valid rank 28 Strassen-based solution
    # and then try to optimize to lower rank using random search.
    
    # Strassen 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2(A, B):
        """Returns 7 products for 2x2*2x2 multiplication."""
        A00, A01, A10, A11 = A[0,0], A[0,1], A[1,0], A[1,1]
        B00, B01, B10, B11 = B[0,0], B[0,1], B[1,0], B[1,1]
        
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        return [[C00, C01], [C10, C11]]
    
    # Build rank 28 Strassen-based decomposition
    # A is 2x4: A = [A0 | A1] where A0, A1 are 2x2
    # B is 4x4: B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [A0*B00 + A1*B10 | A0*B01 + A1*B11]
    
    # For C_left = A0*B00 + A1*B10:
    # Compute X = A0*B00 using Strassen (7 products)
    # Compute Y = A1*B10 using Strassen (7 products)
    # C_left = X + Y
    
    # Similarly for C_right.
    # Total: 4 Strassen computations = 28 products.
    
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28), dtype=float)
    V28 = np.zeros((m * p, rank_28), dtype=float)
    W28 = np.zeros((p * n, rank_28), dtype=float)
    
    # Strassen coefficients for 2x2*2x2:
    # Input A: [[a00, a01], [a10, a11]] -> flattened to [a00, a01, a10, a11]
    # Input B: [[b00, b01], [b10, b11]] -> flattened to [b00, b01, b10, b11]
    # Output C: [[c00, c01], [c10, c11]] -> flattened to [c00, c01, c10, c11]
    
    # Strassen products:
    # M1: (a00+a11)*(b00+b11)
    # M2: (a10+a11)*b00
    # M3: a00*(b01-b11)
    # M4: a11*(b10-b00)
    # M5: (a00+a01)*b11
    # M6: (a10-a00)*(b00+b01)
    # M7: (a01-a11)*(b10+b11)
    
    # Output:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    strassen_U = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1],  # M7: a01-a11
    ])
    
    strassen_V = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 1, 0, -1],  # M3: b01-b11
        [0, 0, 1, -1],  # M4: b10-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1],   # M7: b10+b11
    ])
    
    strassen_W = np.array([
        [1, 0, 0, 0],   # c00 from M1
        [0, 0, 0, 1],   # c10 from M4... wait
    ])
    
    # Let me redo W properly:
    # c00 = M1 + M4 - M5 + M7 -> W for c00: [1, 0, 0, 1, -1, 0, 1]
    # c01 = M3 + M5 -> W for c01: [0, 0, 1, 0, 1, 0, 0]
    # c10 = M2 + M4 -> W for c10: [0, 1, 0, 1, 0, 0, 0]
    # c11 = M1 - M2 + M3 + M6 -> W for c11: [1, -1, 1, 0, 0, 1, 0]
    
    strassen_W = np.array([
        [1, 0, 0, 0],    # c00
        [0, 0, 0, 1],    # c01... no
    ])
    
    # W maps from 7 products to 4 outputs:
    # W is 4x7:
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],    # c01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # c10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0],   # c11 = M1 - M2 + M3 + M6
    ])
    
    # Now build the full rank-28 decomposition:
    # A is 2x4, flattened to 8 elements: [A00, A01, A10, A11, A20, A21, A30, A31]
    # where A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1], A20=A[0,2], etc.
    # Wait, A is 2x4: rows 0,1; cols 0,1,2,3.
    # A0 = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A1 = A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    
    # B is 4x4:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # C is 2x4:
    # C[:, 0:2] = A0*B00 + A1*B10
    # C[:, 2:4] = A0*B01 + A1*B11
    
    # For each Strassen computation, we have 7 products.
    # 4 Strassen computations: 
    #   S1: A0*B00 (7 products)
    #   S2: A1*B10 (7 products)
    #   S3: A0*B01 (7 products)
    #   S4: A1*B11 (7 products)
    
    # Total rank: 28.
    
    # U: 8 x 28
    # V: 16 x 28
    # W: 8 x 28 (C is 2x4, flattened to 8)
    
    U28 = np.zeros((8, 28), dtype=float)
    V28 = np.zeros((16, 28), dtype=float)
    W28 = np.zeros((8, 28), dtype=float)
    
    # S1: A0*B00
    # A0 indices in A: [0,1,2,3] (A[0,0], A[0,1], A[1,0], A[1,1])
    # B00 indices in B: [0,1,4,5] (B[0,0], B[0,1], B[1,0], B[1,1])
    # C_left_top: C[0,0], C[0,1] -> indices 0,1 in C
    # C_left_bottom: C[1,0], C[1,1] -> indices 4,5 in C
    
    for r in range(7):
        col = r  # S1 uses columns 0-6
        U28[[0,1,2,3], col] = strassen_U[r, :]
        V28[[0,1,4,5], col] = strassen_V[r, :]
        W28[[0,1,4,5], col] = strassen_W[:, r]
    
    # S2: A1*B10
    # A1 indices in A: [4,5,6,7] (A[0,2], A[0,3], A[1,2], A[1,3])
    # B10 indices in B: [8,9,12,13] (B[2,0], B[2,1], B[3,0], B[3,1])
    # C_left_top: C[0,0], C[0,1] -> indices 0,1
    # C_left_bottom: C[1,0], C[1,1] -> indices 4,5
    
    for r in range(7):
        col = 7 + r  # S2 uses columns 7-13
        U28[[4,5,6,7], col] = strassen_U[r, :]
        V28[[8,9,12,13], col] = strassen_V[r, :]
        W28[[0,1,4,5], col] = strassen_W[:, r]
    
    # S3: A0*B01
    # A0 indices in A: [0,1,2,3]
    # B01 indices in B: [2,3,6,7] (B[0,2], B[0,3], B[1,2], B[1,3])
    # C_right_top: C[0,2], C[0,3] -> indices 2,3
    # C_right_bottom: C[1,2], C[1,3] -> indices 6,7
    
    for r in range(7):
        col = 14 + r  # S3 uses columns 14-20
        U28[[0,1,2,3], col] = strassen_U[r, :]
        V28[[2,3,6,7], col] = strassen_V[r, :]
        W28[[2,3,6,7], col] = strassen_W[:, r]
    
    # S4: A1*B11
    # A1 indices in A: [4,5,6,7]
    # B11 indices in B: [10,11,14,15] (B[2,2], B[2,3], B[3,2], B[3,3])
    # C_right_top: C[0,2], C[0,3] -> indices 2,3
    # C_right_bottom: C[1,2], C[1,3] -> indices 6,7
    
    for r in range(7):
        col = 21 + r  # S4 uses columns 21-27
        U28[[4,5,6,7], col] = strassen_U[r, :]
        V28[[10,11,14,15], col] = strassen_V[r, :]
        W28[[2,3,6,7], col] = strassen_W[:, r]
    
    # Verify the rank-28 decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the tensor decomposition is correct."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Check dimensions
        if U.shape != (n*m, rank):
            return False
        if V.shape != (m*p, rank):
            return False
        if W.shape != (p*n, rank):
            return False
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, the tensor is T[i,j,k] = 1, and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # So C[i,k] = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
        #           = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    a_part = 0
                    for j in range(m):
                        a_part += U[i*m + j, r] * A[i, j]
                    b_part = 0
                    for j in range(m):
                        b_part += V[j*p + k, r] * B[j, k]
                    C_computed[i, k] += a_part * b_part * W[k*n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    
    if not verify(U28, V28, W28, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank using random search
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27, 26, ... down to 14
    for target_rank in range(27, 13, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random perturbation and projection
        np.random.seed(42 + target_rank)
        
        # Start from rank-28 and try to zero out some columns
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            U_try = U28[:, :target_rank].copy()
            V_try = V28[:, :target_rank].copy()
            W_try = W28[:, :target_rank].copy()
            
            # Add random noise to allow adjustment
            noise_scale = 0.1
            U_try += np.random.randn(*U_try.shape) * noise_scale
            V_try += np.random.randn(*V_try.shape) * noise_scale
            W_try += np.random.randn(*W_try.shape) * noise_scale
            
            if verify(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
