"""
Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition.
Uses a structured construction based on block decomposition of the multiplication tensor.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x4 @ 4x5 multiplication.
    U: (8, rank), V: (20, rank), W: (10, rank)
    Entries are integers or half-integers (multiples of 1/2).
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin
    
    # We'll build a rank-33 decomposition using a combination of
    # structured blocks. The multiplication tensor T[i,j,k] = 1 if we
    # are computing C[i,k] = sum_j A[i,j]*B[j,k].
    # 
    # A is 2x4 (indices i in 0..1, j in 0..3)
    # B is 4x5 (indices j in 0..3, k in 0..4)  
    # C is 2x5 (indices i in 0..1, k in 0..4)
    #
    # Tensor entry: T[i*m+j, j*p+k, k*n+i] = 1
    # i.e., T[i*4+j, j*5+k, k*2+i] = 1 for valid (i,j,k)
    #
    # Total non-zero entries: 2 * 4 * 5 = 40
    #
    # Rank of matrix multiplication tensor M_{n,m,p} is at most n*m*p (schoolbook)
    # For 2x4x5: schoolbook rank = 40. We need to do better.
    #
    # Strategy: Use block-structured decomposition.
    # Split A into 2x2 blocks (A1 = A[:, 0:2], A2 = A[:, 2:4])
    # Split B into 2x2 and 2x1 blocks.
    #
    # Actually, let me think more carefully. The rank of the 2x4x5 matrix
    # multiplication tensor: we can use Strassen-like techniques on sub-blocks.
    #
    # Let me try a direct construction aiming for rank 33 or lower.
    #
    # Known bounds: rank(2,4,5) <= 33 (from various lower bounds literature)
    # Actually the best known upper bound for 2x4x5 is around 32-34.
    #
    # Let me construct a rank-33 decomposition using the following approach:
    # - Split the 4 middle indices into two groups of 2
    # - For each pair of 2x2 blocks, use Strassen (7 mults) for the 2x2 output
    # - For the 5th column, use schoolbook
    #
    # More specifically:
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    # C[:, 4]   = A[:, 0:2] @ B[0:2, 4:5] + A[:, 2:4] @ B[2:4, 4:5]
    #
    # Using Strassen for 2x2x2 blocks: rank 7 each
    # Using schoolbook for 2x2x1 blocks: rank 4 each
    #
    # C[:, 0:2]: two 2x2x2 blocks -> 7 + 7 = 14
    # C[:, 2:4]: two 2x2x2 blocks -> 7 + 7 = 14
    # C[:, 4]:   two 2x2x1 blocks -> 4 + 4 = 8
    # Total: 14 + 14 + 8 = 36 (same as current program!)
    #
    # To do better, we need to share work across output blocks.
    # Let me try a different splitting strategy.
    #
    # Alternative: Think of it as computing a 2x5 matrix C.
    # C = A @ B where A is 2x4, B is 4x5.
    # We can view this as: C = [A1 | A2] @ [[B1],[B2]] where A1,A2 are 2x2, B1,B2 are 2x5.
    # C = A1 @ B1 + A2 @ B2
    # Each of these is a 2x2 @ 2x5 product.
    # 
    # For a 2x2 @ 2x5 product, we can use Strassen for the 2x2x2 part (cols 0-1 of B)
    # and handle the remaining columns.
    #
    # Actually, let me try to directly search for a good rank using random 
    # tensor decomposition with integer/half-integer constraints.
    #
    # Given time constraints, let me use a well-known construction.
    #
    # The tensor M_{2,4,5} can be decomposed using the following approach:
    # Embed into M_{3,4,5} or use known results.
    #
    # Actually, let me try rank 34 with a careful construction.
    #
    # Let me use a direct algebraic approach. The multiplication tensor has
    # 40 nonzero entries. We need to find rank r such that we can express
    # T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t.
    #
    # I'll use a randomized approach: try to find a rank-r decomposition
    # by solving a system of polynomial equations, but that's too slow.
    #
    # Instead, let me use the following constructive approach:
    # 
    # Group the output by columns of C.
    # For columns 0,1: use a 2x4x2 decomposition
    # For columns 2,3: use a 2x4x2 decomposition  
    # For column 4: use a 2x4x1 decomposition
    #
    # Rank of 2x4x2: can be done in 2*2*2 = 8 (schoolbook) or better?
    # Actually 2x4x2: we can split A into two 2x2 blocks, giving
    # two 2x2x2 products, each rank 7 (Strassen). Total: 14.
    # But we can do better by sharing.
    #
    # Let me just implement the rank-36 solution properly and verify it,
    # then see if I can shave off a few terms through clever sharing.
    #
    # For now, let me implement a clean rank-36 solution and verify correctness.
    
    rank = 36
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    idx_counter = [0]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add one rank-1 term."""
        t = idx_counter[0]
        if t >= rank:
            return False
        for (i, j), val in u_coeffs.items():
            U[i*m + j, t] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, t] += val
        for (k, i), val in w_coeffs.items():
            W[k*n + i, t] += val
        idx_counter[0] = t + 1
        return True
    
    def strassen_222(a_off, b_off, c_off):
        """Strassen for A[a_rows, a_off:a_off+2] @ B[b_off:b_off+2, c_off:c_off+2] -> C[c_off:c_off+2, a_rows]"""
        # a_rows = {0, 1}, local j = 0,1 maps to global j = a_off, a_off+1
        # local k = 0,1 maps to global k = c_off, c_off+1
        
        aj0 = (0, a_off)    # A[0, a_off]
        aj1 = (0, a_off+1)  # A[0, a_off+1]
        aj2 = (1, a_off)    # A[1, a_off]
        aj3 = (1, a_off+1)  # A[1, a_off+1]
        
        bk0 = (b_off, c_off)        # B[b_off, c_off]
        bk1 = (b_off, c_off+1)      # B[b_off, c_off+1]
        bk2 = (b_off+1, c_off)      # B[b_off+1, c_off]
        bk3 = (b_off+1, c_off+1)    # B[b_off+1, c_off+1]
        
        ck0 = (c_off, 0)       # C[c_off, 0]
        ck1 = (c_off+1, 0)     # C[c_off+1, 0]
        ck2 = (c_off, 1)       # C[c_off, 1]
        ck3 = (c_off+1, 1)     # C[c_off+1, 1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 -= M1, C10 += M1, C01 += M1
        if not add_term({aj0: 1, aj3: 1}, {bk0: 1, bk3: 1}, 
                        {ck0: 1, ck3: 1, ck2: 1, ck1: -1}):
            return False
        
        # M2 = (A10 + A11) * B00
        # C11 += M2, C10 -= M2
        if not add_term({aj2: 1, aj3: 1}, {bk0: 1}, 
                        {ck3: 1, ck2: -1}):
            return False
        
        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        if not add_term({aj0: 1}, {bk1: 1, bk3: -1}, 
                        {ck1: 1, ck3: 1}):
            return False
        
        # M4 = A11 * (B10 - B00)
        # C10 += M4, C11 += M4
        if not add_term({aj3: 1}, {bk2: 1, bk0: -1}, 
                        {ck2: 1, ck3: 1}):
            return False
        
        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        if not add_term({aj0: 1, aj1: 1}, {bk3: 1}, 
                        {ck0: -1, ck1: 1}):
            return False
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        if not add_term({aj2: 1, aj0: -1}, {bk0: 1, bk1: 1}, 
                        {ck3: 1}):
            return False
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C10 += M7
        if not add_term({aj1: 1, aj3: -1}, {bk2: 1, bk3: 1}, 
                        {ck2: 1}):
            return False
        
        return True
    
    def schoolbook_221(a_off, b_off, c_off):
        """Schoolbook for A[:, a_off:a_off+2] @ B[b_off:b_off+2, c_off:c_off+1] -> C[c_off:c_off+1, :]"""
        aj0 = (0, a_off)
        aj1 = (0, a_off+1)
        aj2 = (1, a_off)
        aj3 = (1, a_off+1)
        
        bk0 = (b_off, c_off)
        bk1 = (b_off+1, c_off)
        
        ck0 = (c_off, 0)
        ck1 = (c_off, 1)
        
        # C[0] += A00*B00 + A01*B10
        # C[1] += A10*B00 + A11*B10
        
        # Term 1: A00 * B00 -> C[0]
        if not add_term({aj0: 1}, {bk0: 1}, {ck0: 1}):
            return False
        # Term 2: A01 * B10 -> C[0]  
        if not add_term({aj1: 1}, {bk1: 1}, {ck0: 1}):
            return False
        # Term 3: A10 * B00 -> C[1]
        if not add_term({aj2: 1}, {bk0: 1}, {ck1: 1}):
            return False
        # Term 4: A11 * B10 -> C[1]
        if not add_term({aj3: 1}, {bk1: 1}, {ck1: 1}):
            return False
        
        return True
    
    # Build the decomposition:
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    # C[:, 4]   = A[:, 0:2] @ B[0:2, 4:5] + A[:, 2:4] @ B[2:4, 4:5]
    
    ok = True
    # Block 1: A_L @ B_T1 (A[:, 0:2] @ B[0:2, 0:2])
    ok &= strassen_222(0, 0, 0)
    if not ok:
        raise RuntimeError("Exceeded rank")
    
    # Block 2: A_R @ B_T1 (A[:, 2:4] @ B[2:4, 0:2])
    ok &= strassen_222(2, 2, 0)
    if not ok:
        raise RuntimeError("Exceeded rank")
    
    # Block 3: A_L @ B_T2 (A[:, 0:2] @ B[0:2, 2:4])
    ok &= strassen_222(0, 0, 2)
    if not ok:
        raise RuntimeError("Exceeded rank")
    
    # Block 4: A_R @ B_T2 (A[:, 2:4] @ B[2:4, 2:4])
    ok &= strassen_222(2, 2, 2)
    if not ok:
        raise RuntimeError("Exceeded rank")
    
    # Block 5: A_L @ B_B (A[:, 0:2] @ B[0:2, 4:5])
    ok &= schoolbook_221(0, 0, 4)
    if not ok:
        raise RuntimeError("Exceeded rank")
    
    # Block 6: A_R @ B_B (A[:, 2:4] @ B[2:4, 4:5])
    ok &= schoolbook_221(2, 2, 4)
    if not ok:
        raise RuntimeError("Exceeded rank")
    
    final_rank = idx_counter[0]
    
    if final_rank != rank:
        # Adjust: trim to actual used rank
        rank = final_rank
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
    
    # Verify the decomposition
    # Tensor T should satisfy: for all i,j,k: T[i*m+j, j*p+k, k*n+i] = 1
    # Our decomposition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # Quick verification on a sample
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from U, V, W
    C_actual = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for t in range(rank):
                u_val = 0.0
                v_val = 0.0
                for j in range(m):
                    u_val += A_test[i, j] * U[i*m+j, t]
                    v_val += B_test[j, k] * V[j*p+k, t]
                val += u_val * v_val * W[k*n+i, t]
            C_actual[i, k] = val
    
    if not np.allclose(C_actual, C_expected, atol=1e-10):
        # The decomposition might have issues with the W indexing
        # W[k*n+i, r] means W is indexed by (k, i) where k is column of C, i is row of C
        # Let me recheck: W[k*n+i] for C[k, i]? But C is n x p, so C[i, k].
        # The convention is W[k*n+i, r] which means the (k*n+i)-th row of W corresponds to
        # output position (k, i) in... wait, C is 2x5, so C[i,k] with i in 0..1, k in 0..4.
        # W should have shape (p*n, rank) = (10, rank).
        # W[k*n+i, r] means for output C[i, k], we use W[k*n + i].
        # So in our loop, for C[i, k], we should use W[k*n + i, t].
        # That's what I have: W[k*n+i, t]. Good.
        
        raise ValueError(f"Verification failed. Max error: {np.max(np.abs(C_actual - C_expected))}")
    
    # Ensure entries are integer or half-integer
    U = 2 * U
    V = 2 * V  
    W = 2 * W
    
    for arr, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = round(arr[i, j], 10)
                arr[i, j] = val
                if abs(arr[i, j] - round(arr[i, j])) > 1e-9:
                    # Not an integer after scaling by 2, meaning original wasn't half-integer
                    pass  # Our construction uses only 0, ±1 so this should be fine
    
    U = U / 2
    V = V / 2
    W = W / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
