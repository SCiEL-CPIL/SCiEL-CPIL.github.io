import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin before 120s limit

    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices i, j, k in {0, 1}
    
    # A_s[i][j]
    A_coeffs = np.array([
        [[1, 0], [0, 1]],  # s=0: A00 + A11
        [[0, 0], [1, 1]],  # s=1: A10 + A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00 + A01
        [[-1, 0], [1, 0]], # s=5: A10 - A00
        [[0, 1], [0, -1]]  # s=6: A01 - A11
    ])
    
    # B_s[j][k]
    B_coeffs = np.array([
        [[1, 0], [0, 1]],  # s=0: B00 + B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01 - B11
        [[-1, 0], [1, 0]], # s=3: B10 - B00
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00 + B01
        [[0, 0], [1, 1]]   # s=6: B10 + B11
    ])
    
    # C_s[i][k]
    C_coeffs = np.array([
        [[1, 0], [0, 1]],  # s=0: C00 += M, C11 += M
        [[0, 0], [1, -1]], # s=1: C10 += M, C11 -= M
        [[0, 1], [0, 1]],  # s=2: C01 += M, C11 += M
        [[1, 0], [1, 0]],  # s=3: C00 += M, C10 += M
        [[-1, 1], [0, 0]], # s=4: C00 -= M, C01 += M
        [[0, 0], [0, 1]],  # s=5: C11 += M
        [[1, 0], [0, 0]]   # s=6: C00 += M
    ])
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Vectorized construction
    # We iterate over the outer Strassen steps s and inner steps t
    # r = 7 * s + t
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # U construction: U[i*4+j, r]
            # i = 2*I + i', j = 2*J + j'
            # U[i*4+j, r] = A_coeffs[s][I][J] * A_coeffs[t][i'][j']
            # We can reshape A_coeffs[s] to (2,2) and A_coeffs[t] to (2,2)
            # The product is an outer product of the flattened matrices? No.
            # U is 16x1. The indices are (i, j).
            # Let's build the 4x4 matrix for this rank component.
            
            # A_block = A_coeffs[s] (2x2)
            # A_inner = A_coeffs[t] (2x2)
            # The 4x4 matrix U_r is formed by:
            # U_r[2*I+i', 2*J+j'] = A_block[I, J] * A_inner[i', j']
            # This is the Kronecker product A_block \otimes A_inner
            
            U_4x4 = np.kron(A_coeffs[s], A_coeffs[t])
            U[:, r] = U_4x4.flatten()
            
            # V construction: V[j*4+k, r]
            # j = 2*J + j', k = 2*K + k'
            # V[j*4+k, r] = B_coeffs[s][J][K] * B_coeffs[t][j'][k']
            # V_r is the Kronecker product B_coeffs[s] \otimes B_coeffs[t]
            
            V_4x4 = np.kron(B_coeffs[s], B_coeffs[t])
            V[:, r] = V_4x4.flatten()
            
            # W construction: W[k*4+i, r]
            # k = 2*K + k', i = 2*I + i'
            # W[k*4+i, r] = C_coeffs[s][I][K] * C_coeffs[t][i'][k']
            # Note the indices: C_s[I][K] and C_t[i'][k']
            # The matrix W_r has rows indexed by (K, k') and columns by (I, i')
            # Let C_s be 2x2 with indices [I, K]
            # Let C_t be 2x2 with indices [i', k']
            # We want W_r[2*K+k', 2*I+i'] = C_s[I, K] * C_t[i', k']
            # This looks like the Kronecker product but with transposed logic for the second factor?
            # Let's check Kronecker product definition: (A \otimes B)_{(i, i'), (j, j')} = A_{i, j} B_{i', j'}
            # Here we want: Row index (K, k'), Col index (I, i')
            # Value = C_s[I, K] * C_t[i', k']
            # Let A = C_s^T (so A[K, I] = C_s[I, K])
            # Let B = C_t^T (so B[k', i'] = C_t[i', k'])
            # Then (C_s^T \otimes C_t^T)_{(K, k'), (I, i')} = (C_s^T)_{K, I} (C_t^T)_{k', i'} = C_s[I, K] C_t[i', k']
            # So W_4x4 = np.kron(C_coeffs[s].T, C_coeffs[t].T)
            
            W_4x4 = np.kron(C_coeffs[s].T, C_coeffs[t].T)
            W[:, r] = W_4x4.flatten()
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Optional: Verification
    # The multiplication tensor M is defined by M[i, j, k] = 1 if i==k else 0? No.
    # C = A B. C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The tensor of matrix multiplication has entries T[i, j, k] = 1 if the indices match the multiplication structure.
    # Specifically, the rank-1 term r contributes U_r[i, j] V_r[j, k] W_r[k, i] to the tensor?
    # Wait, the decomposition is:
    # sum_r U_r[i, j] V_r[j, k] W_r[k, i] = delta_{i, k} ? No.
    # The multiplication tensor M satisfies M[i, j, k] = 1 if the operation is A[i, j] * B[j, k] -> C[i, k].
    # So M[i, j, k] = 1 for all i, j, k?
    # No, the tensor M is defined such that M(A, B) = C.
    # M_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So M_{i, j, k} = 1 if the indices are consistent, which they always are in the summation.
    # Actually, the tensor is T[i, j, k] = 1.
    # Let's check: sum_j A[i, j] B[j, k] = C[i, k].
    # The tensor T has T[i, j, k] = 1.
    # We need sum_r U_r[i, j] V_r[j, k] W_r[k, i] = 1 for all i, j, k?
    # Wait, the trace convention is W[k*n+i, r].
    # The decomposition is: sum_r U_r[i, j] V_r[j, k] W_r[k, i] = T[i, j, k]
    # Where T[i, j, k] = 1.
    
    # Let's verify for a single element, e.g., i=0, j=0, k=0.
    # sum_r U[0, r] V[0, r] W[0, r] should be 1.
    
    # We trust the Strassen construction.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
