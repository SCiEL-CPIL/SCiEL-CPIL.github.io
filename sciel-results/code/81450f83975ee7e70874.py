import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Flattened A: [a, b, c, d] -> indices 0, 1, 2, 3
    # Flattened B: [e, f, g, h] -> indices 0, 1, 2, 3
    # Flattened C: [c11, c12, c21, c22] -> indices 0, 1, 2, 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # c11 += P1, c22 -= P1? No, c22 = P1 - P2 + P3 + P6.
    # Let's define the W coefficients carefully.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Term 0 (P1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C11
    W[3, 0] = 1 # C22
    
    # Term 1 (P2 = (c+d)e)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    W[3, 1] = -1 # C22
    
    # Term 2 (P3 = a(f-h))
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12
    W[3, 2] = 1 # C22
    
    # Term 3 (P4 = d(g-e))
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C11
    W[2, 3] = 1 # C21
    
    # Term 4 (P5 = (a+b)h)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1 # C12
    
    # Term 5 (P6 = (c-a)(e+f))
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # Term 6 (P7 = (b-d)(g+h))
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11
    
    return U, V, W

def permute_4x4(U, V, W):
    # U, V, W are 16x49, 16x49, 16x49 (from kron)
    # We need to reorder rows to match standard flattening.
    # Kron index: (i1*2 + j1)*4 + (i2*2 + j2)
    # Target index: (i1*2 + i2)*4 + (j1*2 + j2)
    
    # Map Kron Index -> Target Index
    # i1, i2, j1, j2 in 0..1
    # Kron: k = (i1*2 + j1)*4 + (i2*2 + j2)
    # Target: t = (i1*2 + i2)*4 + (j1*2 + j2)
    
    perm = np.zeros(16, dtype=int)
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    k = (i1*2 + j1)*4 + (i2*2 + j2)
                    t = (i1*2 + i2)*4 + (j1*2 + j2)
                    perm[k] = t
                    
    # To reorder rows such that row t in new matrix is row k in old matrix?
    # We want new_matrix[t, :] = old_matrix[k, :]
    # So new_matrix = old_matrix[perm_inv, :]?
    # perm[k] = t means row k goes to t.
    # So new_matrix[perm[k], :] = old_matrix[k, :]
    # This is equivalent to new_matrix = old_matrix[perm, :] if perm is a permutation vector where perm[i] is the source index for destination i?
    # No.
    # If perm[k] = t, then destination t gets source k.
    # So new_matrix[t] = old_matrix[k].
    # This is `new_matrix = old_matrix[np.argsort(perm)]`?
    # Let's just construct the new matrix explicitly to be safe.
    
    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)
    
    for k in range(16):
        t = perm[k]
        U_new[t, :] = U[k, :]
        V_new[t, :] = V[k, :]
        W_new[t, :] = W[k, :]
        
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    U2, V2, W2 = get_strassen_2x2x2()
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    U4, V4, W4 = permute_4x4(U4_kron, V4_kron, W4_kron)
    
    # Part 2: 4x4x3 using Trivial (Rank 12)
    # A (4x4) * B2 (4x3) = C2 (4x3)
    # Rank 12.
    U3 = np.zeros((16, 12))
    V3 = np.zeros((12, 12)) # B2 is 4x3, flattened 12
    W3 = np.zeros((12, 12)) # C2 is 4x3, flattened?
    # Task W index: k*n + i.
    # For C2, i in 0..3, k in 0..2.
    # Index in W3: k*4 + i.
    # Total 12 rows.
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # U index: i*4 + j
                U3[i*4 + j, col] = 1
                # V3 index: j*3 + k (B2 is 4x3, row j, col k)
                V3[j*3 + k, col] = 1
                # W3 index: k*4 + i (C2 is 4x3, row i, col k -> W index k*n+i)
                W3[k*4 + i, col] = 1
                col += 1
                
    # Combine
    # U total: 16 x (49 + 12) = 16 x 61
    # V total: 28 x 61
    #   Rows 0-15: B1 (4x4)
    #   Rows 16-27: B2 (4x3)
    # W total: 28 x 61
    #   Rows 0-15: C1 (4x4) -> W index k*4+i for k,i in 0..3
    #   Rows 16-27: C2 (4x3) -> W index 16 + k*4+i for k in 0..2, i in 0..3
    
    R = 61
    U = np.zeros((16, R))
    V = np.zeros((28, R))
    W = np.zeros((28, R))
    
    # Fill Part 1
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    # Fill Part 2
    U[:, 49:] = U3
    V[16:, 49:] = V3
    W[16:, 49:] = W3
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def tensor_residual(U, V, W, n, m, p):
    r = U.shape[1]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i*m + j
                J = j*p + k
                K = k*n + i
                error += (T[I, J, K] - 1)**2
    for I in range(n*m):
        for J in range(m*p):
            for K in range(p*n):
                i = I // m
                j = I % m
                j2 = J // p
                k = J % p
                k2 = K // n
                i2 = K % n
                if i == i2 and j == j2 and k == k2:
                    continue
                error += T[I, J, K]**2
    return error

def check_solution(U, V, W, n, m, p):
    r = U.shape[1]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i*m + j
                J = j*p + k
                K = k*n + i
                if abs(T[I, J, K] - 1) > 1e-9:
                    return False
    for I in range(n*m):
        for J in range(m*p):
            for K in range(p*n):
                i = I // m
                j = I % m
                j2 = J // p
                k = J % p
                k2 = K // n
                i2 = K % n
                if i == i2 and j == j2 and k == k2:
                    continue
                if abs(T[I, J, K]) > 1e-9:
                    return False
    return True

def find_decomp(n, m, p, target_rank, max_time=50):
    # This function is no longer used in the main construct, 
    # but kept to satisfy potential interface requirements if any.
    return None
