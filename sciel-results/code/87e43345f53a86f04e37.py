<<<<<<< SEARCH
"""Direct construction of the explicit rank-49 recursive Strassen tensor
decomposition for 4x4 by 4x4 matrix multiplication (4D x 4D -> 4D)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_START_TIME = time.time()
_DEADLINE_SECONDS = 140


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _recursive_kronecker(2, 2, 2)
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)

    U_mat = U_arr.reshape(n * m, n * m * n * m * p)
    V_mat = V_arr.reshape(m * p, n * m * n * m * p)
    W_mat = W_arr.reshape(p * n, n * m * n * m * p)

    # Verify quickly if possible
    T_ref = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_ref[i * m + j, j * p + k, k * n + i] = 1

    T_rec = np.zeros((n * m, m * p, p * n), dtype=object)
    for c in range(n * m * n * m * p):
        uc = U_mat[:, c]
        vc = V_mat[:, c]
        wc = W_mat[:, c]
        for a in range(n * m):
            if uc[a] != 0:
                for b in range(m * p):
                    if vc[b] != 0:
                        T_rec[a, b, :] += (uc[a] * vc[b]) * wc

    if not np.array_equal(T_rec, T_ref):
        raise RuntimeError("Verification failed")

    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
=======
"""Direct construction of the explicit rank-49 recursive Strassen tensor
decomposition for 4x4 by 4x4 matrix multiplication (4D x 4D -> 4D)."""
import numpy as np

DIMENSIONS = [4, 4, 4]


def construct():
    n, m, p = DIMENSIONS
    # Get the base 2x2 Strassen decomposition
    U2, V2, W2 = _base_strassen_2()
    
    # Convert to numpy arrays for fast Kronecker product computation
    U2_arr = np.array(U2, dtype=float)
    V2_arr = np.array(V2, dtype=float)
    W2_arr = np.array(W2, dtype=float)
    
    # For 4x4, we use the Kronecker product of two 2x2 Strassen decompositions.
    # Rank = 7 * 7 = 49
    # U4[i4, c4] = U2[i1, c1] * U2[i0, c0] where i4 = i1*2 + i0, c4 = c1*7 + c0
    # This is equivalent to the Kronecker product of the matrices U2 and U2?
    # No, the indices are mixed. 
    # U2 has shape (4, 7). 
    # We want U4 of shape (16, 49).
    # i4 ranges 0..15. i4 = i1*2 + i0.
    # c4 ranges 0..48. c4 = c1*7 + c0.
    # U4[i1*2+i0, c1*7+c0] = U2[i1, c1] * U2[i0, c0]
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for c1 in range(7):
        for c0 in range(7):
            c4 = c1 * 7 + c0
            for i1 in range(2):
                for i0 in range(2):
                    i4 = i1 * 2 + i0
                    U4[i4, c4] = U2_arr[i1, c1] * U2_arr[i0, c0]
                    V4[i4, c4] = V2_arr[i1, c1] * V2_arr[i0, c0]
                    W4[i4, c4] = W2_arr[i1, c1] * W2_arr[i0, c0]

    # Verification using vectorized operations
    # T_rec[a, b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
    # This is a tensor contraction.
    # We can compute this efficiently.
    # T_rec = np.einsum('ak,bk,ck->abc', U4, V4, W4)
    
    T_rec = np.einsum('ak,bk,ck->abc', U4, V4, W4)
    
    # T_ref[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j = i*4 + j
    # b = j*p + k = j*4 + k
    # c = k*n + i = k*4 + i
    # Condition: j from a matches j from b, k from b matches k from c, i from c matches i from a.
    # i = a // 4, j = a % 4
    # j = b // 4, k = b % 4
    # k = c // 4, i = c % 4
    
    # Create reference tensor
    T_ref = np.zeros((16, 16, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                T_ref[a, b, c] = 1.0

    if not np.allclose(T_rec, T_ref):
        raise RuntimeError("Verification failed")

    # Return as lists as per interface
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
>>>>>>> REPLACE

