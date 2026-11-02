import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Rank-8 decomposition for 2x2x4 matrix multiplication
    # A (2x2) * B (2x4) = C (2x4)
    # U: shape (4, 8) - A factors: U[i*2+j, r]
    # V: shape (8, 8) - B factors: V[j*4+k, r]
    # W: shape (8, 8) - C factors: W[k*2+i, r]

    U24 = np.zeros((4, 8))
    V24 = np.zeros((8, 8))
    W24 = np.zeros((8, 8))

    # 8 bilinear forms for 2x2x4
    # r=0: A[0,0] * B[0,0]
    U24[0, 0] = 1
    V24[0, 0] = 1
    W24[0, 0] = 1

    # r=1: A[0,0] * B[0,1]
    U24[0, 1] = 1
    V24[1, 1] = 1
    W24[1, 0] = 1

    # r=2: A[0,0] * B[1,0]
    U24[0, 2] = 1
    V24[4, 2] = 1
    W24[2, 0] = 1

    # r=3: A[0,0] * B[1,1]
    U24[0, 3] = 1
    V24[5, 3] = 1
    W24[3, 0] = 1

    # r=4: A[1,0] * B[0,0]
    U24[2, 4] = 1
    V24[0, 4] = 1
    W24[0, 1] = 1

    # r=5: A[1,0] * B[0,1]
    U24[2, 5] = 1
    V24[1, 5] = 1
    W24[1, 1] = 1

    # r=6: A[1,0] * B[1,0]
    U24[2, 6] = 1
    V24[4, 6] = 1
    W24[2, 1] = 1

    # r=7: A[1,0] * B[1,1]
    U24[2, 7] = 1
    V24[5, 7] = 1
    W24[3, 1] = 1

    # Verify: this should compute A*B correctly
    # For A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[0,3] = a00*b03 + a01*b13
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    # C[1,3] = a10*b03 + a11*b13

    # Check if this decomposition is valid
    # The tensor T[i,j,k] = 1 if C[i,k] = A[i,j]*B[j,k]
    # Our decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let me verify with random matrices
    A_test = np.array([[1, 2], [3, 4]], dtype=float)
    B_test = np.array([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=float)
    C_test = A_test @ B_test

    # Compute using our factors
    C_comp = np.zeros((2, 4))
    for r in range(8):
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    C_comp[i, k] += U24[i*2+j, r] * V24[j*4+k, r] * W24[k*2+i, r]

    # This simple rank-8 decomposition doesn't work directly because it's just
    # the trivial expansion. Let me use a proper rank-8 algorithm.
    
    # Actually, for 2x2x4, the trivial rank is 8 (each output element needs
    # 2 multiplications, 4 output elements in each row, 2 rows = 8 total).
    # But we can do better with Strassen-like methods.
    
    # Let me use the correct rank-8 decomposition for 2x2x4:
    # This is equivalent to two 2x2x2 multiplications, each with rank 7,
    # but we can share some factors.
    
    # Actually, the minimal rank for 2x2x4 is known to be 8.
    # The trivial decomposition has rank 8:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term A[i,j]*B[j,k] is a rank-1 tensor.
    # There are 2*2*2 = 8 such terms (i in {0,1}, j in {0,1}, k in {0,1,2,3} but
    # B has 4 columns, so j is 0 or 1, and k is 0,1,2,3).
    # Wait, B is 2x4, so j in {0,1}, k in {0,1,2,3}.
    # Number of terms: 2 (i) * 2 (j) * 4 (k) / 2 (since each A[i,j] multiplies
    # all 4 B[j,k]) = 8 terms.
    
    # The trivial rank-8 decomposition:
    # For each (i,j) pair, we have A[i,j] * B[j,0:4]
    # That's 4 terms, each contributing to 4 output elements.
    # Total: 4 terms * 4 outputs = 16 rank-1 contributions, but grouped into 4 groups.
    
    # Let me redefine:
    # r=0: A[0,0] * B[0,:] -> contributes to C[0,0:4]
    # r=1: A[0,1] * B[1,:] -> contributes to C[0,0:4]
    # r=2: A[1,0] * B[0,:] -> contributes to C[1,0:4]
    # r=3: A[1,1] * B[1,:] -> contributes to C[1,0:4]
    
    # But this is only rank 4, which is too good. The issue is that each "term"
    # A[i,j] * B[j,k] for fixed i,j and varying k is not a rank-1 tensor in the
    # full tensor sense. A rank-1 tensor in M_{2,2,4} is of the form
    # u_i * v_j * w_k for all i,j,k.
    
    # So the trivial decomposition has rank 8:
    # T[i,j,k] = 1 if we want C[i,k] = A[i,j]*B[j,k]
    # The tensor is: T[i,j,k] = 1 for all i,j,k where the product contributes.
    # Actually, the matrix multiplication tensor for (2,2,4) is:
    # T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # This is a 2x2x4 tensor with all entries 1.
    # The rank of the all-ones tensor of size 2x2x4 is... 
    # It's the sum of 8 rank-1 tensors: e_i * e_j * e_k for all i,j,k.
    # But we can do better: the all-ones 2x2x4 tensor can be written as
    # (e_0 + e_1) * (e_0 + e_1) * (e_0 + e_1 + e_2 + e_3)
    # which is rank 1! But that's not right because the tensor isn't all ones.
    
    # Let me think again. The matrix multiplication tensor M_{2,2,4} has:
    # M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all i,j,k where j is a valid index connecting A and B.
    # i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # M[i,j,k] = 1 for all 2*2*4 = 8 combinations.
    # This is indeed the all-ones 2x2x4 tensor.
    # The rank of the all-ones tensor of size a x b x c is 1 (it's u*v*w where
    # u=(1,1,...,1), v=(1,1,...,1), w=(1,1,...,1)).
    # But that can't be right for matrix multiplication!
    
    # I'm confusing myself. The matrix multiplication tensor is NOT all ones.
    # The tensor M_{m,n,p} is defined by:
    # M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give
    # a contribution to C[i,k].
    # For standard matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k.
    # For 2x2x4: i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # M[i,j,k] = 1 for all 8 combinations.
    # This IS the all-ones tensor, and its rank is 1.
    
    # But that contradicts the known result that matrix multiplication has
    # non-trivial rank. The issue is that the tensor for matrix multiplication
    # is NOT the all-ones tensor. Let me look this up properly.
    
    # The matrix multiplication tensor M_{m,n,p} is:
    # M = sum_{i=0}^{m-1} sum_{j=0}^{n-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C
    # where e_i^A is the i-th basis vector in the A-space (size m*n),
    # e_j^B is the j-th basis vector in the B-space (size n*p),
    # e_k^C is the k-th basis vector in the C-space (size m*p).
    # But the indices are coupled: for each triple (i,j,k) where the j-index
    # of A matches the j-index of B, we have a term.
    
    # For 2x2x4:
    # A is 2x2, so A-space has indices (i,j) with i in {0,1}, j in {0,1} -> 4 elements
    # B is 2x4, so B-space has indices (j,k) with j in {0,1}, k in {0,1,2,3} -> 8 elements
    # C is 2x4, so C-space has indices (i,k) with i in {0,1}, k in {0,1,2,3} -> 8 elements
    # The tensor M has M[(i,j),(j,k),(i,k)] = 1 for all valid i,j,k.
    # So M has 2*2*4 = 8 non-zero entries, all equal to 1.
    # The tensor is:
    # M = e_{00}^A * e_{00}^B * e_{00}^C + e_{00}^A * e_{01}^B * e_{01}^C + ...
    
    # This tensor has rank 8 in general (the trivial decomposition).
    # But it might have lower rank.
    
    # For now, let me just use the trivial rank-8 decomposition and build
    # the 4x4x8 problem from two 2x2x4 blocks.
    
    # Trivial rank-8 for 2x2x4:
    # U24[i*2+j, r] = 1 if (i,j) corresponds to term r
    # V24[j*4+k, r] = 1 if (j,k) corresponds to term r
    # W24[k*2+i, r] = 1 if (i,k) corresponds to term r
    
    # Reset and use trivial decomposition
    U24 = np.zeros((4, 8))
    V24 = np.zeros((8, 8))
    W24 = np.zeros((8, 8))

    r = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U24[i*2+j, r] = 1
                V24[j*4+k, r] = 1
                W24[k*2+i, r] = 1
                r += 1

    # Verify
    C_comp = np.zeros((2, 4))
    for rr in range(8):
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    C_comp[i, k] += U24[i*2+j, rr] * V24[j*4+k, rr] * W24[k*2+i, rr]

    if not np.allclose(C_comp, C_test):
        # Fallback: use the full 4x4x8 trivial decomposition
        rank = n * m * p  # 4*4*8 = 128
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build 4x4x8 from two 2x2x4 blocks
    # Block 1: rows 0-1 of A, all of B, rows 0-1 of C
    # Block 2: rows 2-3 of A, all of B, rows 2-3 of C
    
    rank = 16  # 2 blocks * 8 rank each
    U = np.zeros((16, rank))  # n*m = 4*4 = 16
    V = np.zeros((32, rank))  # m*p = 4*8 = 32
    W = np.zeros((32, rank))  # p*n = 8*4 = 32

    # Block 1: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    # Wait, this doesn't work directly because the 2x2x4 decomposition assumes
    # A is 2x2 and B is 2x4. For the 4x4x8 problem, A is 4x4 and B is 4x8.
    
    # Let me reconsider. The 4x4x8 problem can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    
    # We can split this into 4 independent 2x2x4 problems:
    # Block (0,0): i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # Block (0,1): i in {0,1}, j in {2,3}, k in {0,1,2,3}
    # Block (1,0): i in {2,3}, j in {0,1}, k in {0,1,2,3}
    # Block (1,1): i in {2,3}, j in {2,3}, k in {0,1,2,3}
    # But this only covers k in {0,1,2,3}, not k in {4,5,6,7}.
    
    # Actually, B is 4x8, so j in {0,1,2,3} and k in {0,1,2,3,4,5,6,7}.
    # We need to cover all k values.
    
    # Let me split differently:
    # For each pair of i-rows and j-rows, we have a 2x2x8 subproblem.
    # 2x2x8 can be decomposed into two 2x2x4 subproblems (k in {0,1,2,3} and k in {4,5,6,7}).
    
    # So we have 4 (i,j) blocks * 2 (k) blocks = 8 subproblems, each 2x2x4 with rank 8.
    # Total rank: 8 * 8 = 64.
    
    # That's still better than 98, but let me try to do better.
    
    # Alternative: split A into 2x2 blocks and B into 2x4 blocks.
    # A is 4x4, so we can split it into four 2x2 blocks:
    # A[0:2, 0:2], A[0:2, 2:4], A[2:4, 0:2], A[2:4, 2:4]
    # B is 4x8, so we can split it into four 2x4 blocks:
    # B[0:2, 0:4], B[0:2, 4:8], B[2:4, 0:4], B[2:4, 4:8]
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j1 in {0,1}} sum_{j0 in {0,1}} A[i, 2*j1+j0] * B[2*j1+j0, k]
    
    # For fixed i-block (i1 in {0,1}) and k-block (k1 in {0,1}):
    # C[2*i1+i0, 2*k1+k0] = sum_{j1 in {0,1}} sum_{j0 in {0,1}} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    # This is a 2x2x4 problem for each (i1, k1) pair, but with j ranging over 4 values.
    # Actually, for fixed i1, k1, the sum is over j1 in {0,1} and j0 in {0,1},
    # so it's a 2x4x4 problem (2 i-values, 4 j-values, 4 k-values).
    
    # Let me just use the straightforward approach: 8 subproblems of 2x2x4, each rank 8.
    # Total rank: 64.
    
    rank = 64
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    r = 0
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                # 2x2x4 subproblem: i in {i1}, j in {j1}, k in {k1}
                # i0 in {0,1}, j0 in {0,1}, k0 in {0,1,2,3}
                for i0 in range(2):
                    for j0 in range(2):
                        for k0 in range(4):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            U[i*4+j, r] = 1
                            V[j*8+k, r] = 1
                            W[k*4+i, r] = 1
                            r += 1

    # Verify
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 8)
    C_test = A_test @ B_test

    C_comp = np.zeros((4, 8))
    for rr in range(rank):
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    C_comp[i, k] += U[i*4+j, rr] * V[j*8+k, rr] * W[k*4+i, rr]

    if not np.allclose(C_comp, C_test):
        # Fallback to trivial
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
