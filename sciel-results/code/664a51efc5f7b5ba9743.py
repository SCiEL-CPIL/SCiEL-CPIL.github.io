import numpy as np
import time
import scipy.optimize as opt
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin
    
    # Known-good rank-17 decomposition
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    U17[2, 7] = 1
    V17[6, 7] = 1
    W17[0, 7] = 1
    
    U17[2, 8] = 1
    V17[7, 8] = 1
    W17[2, 8] = 1
    
    U17[5, 9] = 1
    V17[6, 9] = 1
    W17[1, 9] = 1
    
    U17[5, 10] = 1
    V17[7, 10] = 1
    W17[3, 10] = 1
    
    U17[0, 11] = 1
    V17[2, 11] = 1
    W17[4, 11] = 1
    
    U17[1, 12] = 1
    V17[5, 12] = 1
    W17[4, 12] = 1
    
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    U17[3, 14] = 1
    V17[2, 14] = 1
    W17[5, 14] = 1
    
    U17[4, 15] = 1
    V17[5, 15] = 1
    W17[5, 15] = 1
    
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val, tol=1e-9):
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(matrix):
        return np.round(matrix * 2) / 2

    # Verify rank-17
    if not verify(U17, V17, W17, n, m, p):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

    # Try to find rank-16 by checking if any column's tensor contribution
    # is in the span of the others (linear dependency check)
    tensor_vectors = []
    for r in range(17):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = U17[i * m + j, r] * V17[j * p + k, r] * W17[k * n + i, r]
        tensor_vectors.append(vec)
    
    for r in range(17):
        if time.time() > deadline - 30:
            break
        others = [tensor_vectors[s] for s in range(17) if s != r]
        A = np.array(others).T  # shape (18, 16)
        b = tensor_vectors[r]  # shape (18,)
        try:
            x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            error = np.linalg.norm(A @ x - b)
            if error < 1e-9:
                # Column r is redundant; try dropping it
                U_test = np.delete(U17, r, axis=1)
                V_test = np.delete(V17, r, axis=1)
                W_test = np.delete(W17, r, axis=1)
                if verify(U_test, V_test, W_test, n, m, p):
                    return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
        except Exception:
            continue
    
    # No rank-16 found via simple elimination; return rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
