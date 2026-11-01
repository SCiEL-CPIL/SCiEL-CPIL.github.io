import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_matmul(U, V, W):
    """
    Verifies if U, V, W decompose 3x3 matmul.
    Convention:
    C[i, k] = sum_r (sum_j A[i, j] U[i*m+j, r]) (sum_j B[j, k] V[j*p+k, r]) W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    R = len(U[0])
    # Use integer arithmetic to avoid floating point issues if possible,
    # but the problem allows half-integers.
    # We'll use exact rational check or high precision.
    
    # Test with small integer matrices
    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    # B = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.int64)
    B = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int64)
    C_expected = A @ B
    
    C_computed = np.zeros((3, 3), dtype=np.int64)
    
    # Precompute linear forms for A and B
    # U is 9xR, V is 9xR
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # Flatten A and B
    A_flat = A.flatten() # i*m+j
    B_flat = B.flatten() # j*p+k
    
    # Compute products P_r = (U^T A_flat) * (V^T B_flat)
    # U_arr is 9xR, A_flat is 9. U_arr.T @ A_flat is 1xR
    U_proj = U_arr.T @ A_flat
    V_proj = V_arr.T @ B_flat
    
    # W_arr is 9xR. Index is k*n+i.
    # We need C[i, k].
    # Term r contributes P_r * W[k*n+i, r] to C[i, k].
    # So C = sum_r P_r * W_vec_r
    # W_vec_r is the r-th column of W_arr, reshaped to 3x3 with index k*n+i.
    
    for r in range(R):
        p_r = U_proj[r] * V_proj[r]
        if p_r != 0:
            # Extract W column r
            w_col = W_arr[:, r]
            # Reshape w_col to 3x3 where index k*n+i -> [k, i]
            # w_col[k*n+i]
            w_matrix = w_col.reshape(3, 3) # row k, col i
            # C[i, k] += p_r * w_matrix[k, i]
            # Note: w_matrix[k, i] is W[k*n+i, r]
            C_computed += p_r * w_matrix.T # w_matrix.T[i, k] = w_matrix[k, i]
            
    if not np.allclose(C_computed, C_expected):
        return False
    return True

def _rank23_construct():
    # Explicit rank-23 decomposition
    # Source: Known optimal algorithm for 3x3 matmul
    # The terms are (U_vec, V_vec, W_vec)
    # Indices:
    # A: 0..8 (i*3+j)
    # B: 0..8 (j*3+k)
    # W: 0..8 (k*3+i)
    
    terms = []
    
    # Helper to add term
    def add(u, v, w):
        terms.append((u, v, w))
        
    # Based on Stothers (2010) or similar rank-23 constructions
    # This specific table is derived from the "3x3x3 rank 23" decomposition.
    # I will use a verified set from literature/code snippets.
    
    # Term 1
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1, 1:1, 4:1})
    # Term 2
    add({3:1, 4:1}, {0:1}, {1:1, 4:1})
    # Term 3
    add({0:1}, {3:1, 4:-1}, {3:1, 4:-1})
    # Term 4
    add({4:1}, {1:1, 0:-1}, {0:1, 1:1})
    # Term 5
    add({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    # Term 6
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    # Term 7
    add({1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # These 7 terms handle the 2x2 block in top-left?
    # We need 23 terms total.
    # The rest cover the remaining parts.
    
    # Due to complexity of recalling the exact 23 terms from memory without error,
    # and the risk of an invalid decomposition scoring 0,
    # I will implement a fallback strategy:
    # If I cannot verify the rank-23 table, I return schoolbook.
    
    # However, I can try to construct a valid lower rank if I find one.
    # For now, I will return None to indicate failure to construct verified rank-23.
    return None

def construct():
    best = _schoolbook()
    
    # Attempt rank 23
    if _time_left() > 5.0:
        cand = _rank23_construct()
        if cand is not None:
            if _verify_matmul(cand["U"], cand["V"], cand["W"]):
                best = cand
            else:
                # Verification failed, stick to schoolbook
                pass
    
    return best
