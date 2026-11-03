import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s limit

    # 1. Define Strassen factors for 2x2x2
    # U2[i, j, r], V2[j, k, r], W2[k, i, r]
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # 2. Build 4x4x4 factors using tensor product (Kronecker)
    # Rank 49
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
                    
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
                    
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # 3. Build 4x4x8 factors
    # Strategy: Split B into two 4x4 blocks.
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # We can use the same A factors (U4) for both blocks.
    # However, the rank is the sum of the ranks if we treat them independently,
    # UNLESS we can share the rank-1 terms.
    # In the tensor decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r],
    # the index k goes up to 8.
    # If we set U[i,j, r] = U4[i,j, r//2] for r in 0..97? No, that's not right.
    #
    # Let's look at the structure.
    # For k in 0..3 (first block):
    #   V[j,k,r] = V4[j,k, r] for r in 0..48
    #   W[k,i,r] = W4[k,i, r] for r in 0..48
    #   U[i,j,r] = U4[i,j, r] for r in 0..48
    #
    # For k in 4..7 (second block):
    #   Let k' = k - 4.
    #   We want to contribute to C[i, k] = C[i, 4+k'].
    #   The term is sum_r U[i,j,r] V[j, 4+k', r] W[4+k', i, r].
    #   
    # If we simply duplicate the rank-49 decomposition for the second block,
    # we get rank 98.
    # U[i,j, 49+r] = U4[i,j, r]
    # V[j, 4+k', 49+r] = V4[j, k', r]
    # W[4+k', i, 49+r] = W4[k', i, r]
    #
    # This yields rank 98.
    #
    # Can we do better?
    # The tensor for 4x4x8 is the direct sum of two 4x4x4 tensors.
    # The rank of a direct sum is at most the sum of the ranks.
    # Is it exactly the sum?
    # For matrix multiplication tensors, the rank is known to be additive in this way?
    # Actually, the rank of M_{<n,m,p>} is not known to be additive, but for these small sizes,
    # it is likely that 98 is optimal or close to it.
    #
    # However, we can try to verify if there are linear dependencies.
    # Let's construct the rank-98 solution first.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First block (k=0..3)
    U[:, :49] = U4.reshape(16, 49)
    for j in range(4):
        for k in range(4):
            idx = j * 8 + k
            V[idx, :49] = V4[j, k, :]
            idx2 = k * 4 + i # Wait, W index is k*n + i
            # W has shape (p*n, rank) = (32, rank). Index is k*4 + i.
            for i in range(4):
                W[k * 4 + i, :49] = W4[k, i, :]
                
    # Second block (k=4..7)
    U[:, 49:] = U4.reshape(16, 49)
    for j in range(4):
        for k in range(4):
            idx = j * 8 + (4 + k)
            V[idx, 49:] = V4[j, k, :]
            for i in range(4):
                W[(4 + k) * 4 + i, 49:] = W4[k, i, :]

    # Verify the solution
    if time.time() < deadline - 5:
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails
            U_school = np.zeros((16, 128))
            V_school = np.zeros((32, 128))
            W_school = np.zeros((32, 128))
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        r = i * 32 + j * 8 + k
                        U_school[i*4+j, r] = 1
                        V_school[j*8+k, r] = 1
                        W_school[k*4+i, r] = 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # If verification takes too long or fails, return schoolbook
    U_school = np.zeros((16, 128))
    V_school = np.zeros((32, 128))
    W_school = np.zeros((32, 128))
    for i in range(4):
        for j in range(4):
            for k in range(8):
                r = i * 32 + j * 8 + k
                U_school[i*4+j, r] = 1
                V_school[j*8+k, r] = 1
                W_school[k*4+i, r] = 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
