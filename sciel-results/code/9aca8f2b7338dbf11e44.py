import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin well under 150s

    n, m, p = DIMENSIONS
    RANK = 49  # 7 * 7: outer Strassen (7 terms) x inner Strassen (7 terms)

    # Standard Strassen 2x2 coefficients
    # a_coeffs[s][i][j]: coefficient of A[i][j] in s-th A-form
    # b_coeffs[s][j][k]: coefficient of B[j][k] in s-th B-form
    # c_coeffs[s][i][k]: coefficient of C[i][k] in s-th C-form
    #
    # Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5

    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=np.int32)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=np.int32)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=np.int32)

    # Build U, V, W using Kronecker product structure:
    # The 4x4x4 decomposition is the tensor product of two 2x2x2 Strassen decompositions.
    # For index i = 2*S + i2, j = 2*T + j2:
    #   U[i*4+j, 7*s+t] = a_coeffs[s][S][T] * a_coeffs[t][i2][j2]
    #   V[j*4+k, 7*s+t] = b_coeffs[s][T][K] * b_coeffs[t][j2][k2]
    #   W[k*4+i, 7*s+t] = c_coeffs[s][S][K] * c_coeffs[t][i2][k2]
    #
    # We use np.kron to build these efficiently.
    # U has shape (16, 49) where row = i*4+j and col = 7*s+t
    # This is equivalent to kron(outer_A, inner_A) reshaped appropriately.

    # For U: outer part is a_coeffs reshaped to (4, 7) [row=i*2+j, col=s]
    # inner part is a_coeffs reshaped to (4, 7) [row=i2*2+j2, col=t]
    # Then U = kron(outer, inner) gives (16, 49) but we need to reorder columns.
    
    # Actually, let's be more careful. We want:
    # U[(2S+i2)*4 + (2T+j2), 7*s+t] = a_coeffs[s,S,T] * a_coeffs[t,i2,j2]
    
    # Let's build this using broadcasting.
    # a_coeffs has shape (7, 2, 2)
    # We want for each (s, t): a_coeffs[s] @ a_coeffs[t] but that's not right.
    # We want the outer product structure: for each (s,t), the 4x4 block is 
    #   a_coeffs[s, S, T] * a_coeffs[t, i2, j2] for S,T,i2,j2 in {0,1}
    # which is kron(a_coeffs[s].reshape(2,2).T, a_coeffs[t].reshape(2,2).T) 
    # but with row index = (2S+i2)*4 + (2T+j2) = 4*(2S+i2) + (2T+j2)
    # and a_coeffs[s].reshape(2,2) has [S,T] indexing.
    # So the 4x4 matrix for (s,t) is:
    #   M_{(S,i2),(T,j2)} = a_coeffs[s,S,T] * a_coeffs[t,i2,j2]
    # This is kron(A_s, A_t) where A_s[S,T] = a_coeffs[s,S,T] and A_t[i2,j2] = a_coeffs[t,i2,j2]
    # But kron(A_s, A_t) has row (S,i2) and col (T,j2) with value A_s[S,T]*A_t[i2,j2]? No.
    # kron(A_s, A_t)[(S,i2),(T,j2)] = A_s[S,T] * A_t[i2,j2]. Yes!
    
    # So U[:, 7*s+t] = kron(A_s, A_t).ravel() where A_s = a_coeffs[s].reshape(2,2)
    # and the ravel order gives row = (S*2+i2)*4 + (T*2+j2) = 4*(2S+i2) + (2T+j2). Correct!
    
    # Similarly for V and W.
    
    U = np.zeros((16, RANK), dtype=np.int32)
    V = np.zeros((16, RANK), dtype=np.int32)
    W = np.zeros((16, RANK), dtype=np.int32)
    
    for s in range(7):
        As = a_coeffs[s].reshape(2, 2)
        Bs = b_coeffs[s].reshape(2, 2)
        Cs = c_coeffs[s].reshape(2, 2)
        for t in range(7):
            At = a_coeffs[t].reshape(2, 2)
            Bt = b_coeffs[t].reshape(2, 2)
            Ct = c_coeffs[t].reshape(2, 2)
            
            col = 7 * s + t
            U[:, col] = np.kron(As, At).ravel()
            V[:, col] = np.kron(Bs, Bt).ravel()
            W[:, col] = np.kron(Cs, Ct).ravel()
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

