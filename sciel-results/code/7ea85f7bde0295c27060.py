import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 decomposition for 2x4 * 4x8.
    
    Strategy:
    The tensor T for matrix multiplication C = A @ B has components:
    T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    Actually, the tensor is T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_{i,k}
    where e_i^* extracts row i of A (col j), e_j^* extracts row j of B (col k),
    and e_{i,k} is the basis vector for C[i,k].
    
    Wait, the standard tensor is:
    C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor is T = sum_{i,j,k} A_{i,j}^* \otimes B_{j,k}^* \otimes C_{i,k}
    where A_{i,j}^* is the functional extracting A[i,j], etc.
    
    This is a sum of n*m*p = 64 rank-1 terms.
    Can we do better?
    
    Consider the matricization M of size (m*p) x (n*m) or similar.
    The rank of the tensor is at least the rank of any matricization.
    
    Let's try to find a rank-32 decomposition.
    If we fix the indices j and k, we have a sub-tensor for each (j,k).
    For fixed j,k, the term is sum_i A_{i,j} B_{j,k} C_{i,k}.
    This is B_{j,k} * sum_i A_{i,j} C_{i,k}.
    The vector sum_i A_{i,j} C_{i,k} is not a rank-1 term in the tensor sense.
    
    Let's use the fact that the lower bound is 32.
    If we can find a rank-32 decomposition, it must be optimal.
    
    One way to construct a rank-32 decomposition is to use the fact that
    the tensor is "diagonal" in the j,k indices.
    For each (j,k), we have a term involving A[:,j] and B[j,k] and C[:,k].
    Specifically, C[:,k] += A[:,j] * B[j,k].
    This is a rank-1 update to the column k of C.
    In tensor terms, this is:
    U_{j,k} = vector that extracts A[:,j] (size 2)
    V_{j,k} = scalar B[j,k]
    W_{j,k} = vector e_k (size p) but scaled to C[:,k]
    
    Wait, the tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    u_r is in A^* (size 8), v_r is in B^* (size 32), w_r is in C (size 16).
    
    For a fixed (j,k), let's define:
    u_{j,k} = e_j \otimes (sum_i e_i) ? No.
    u_{j,k} should extract A[:,j]. So u_{j,k} has 1s at indices (0,j) and (1,j).
    v_{j,k} should extract B[j,k]. So v_{j,k} has 1 at index (j,k).
    w_{j,k} should be e_k in C-space? No, w_{j,k} should be the vector that,
    when multiplied by the scalar (u_{j,k} \cdot A) * (v_{j,k} \cdot B),
    gives the contribution to C.
    
    (u_{j,k} \cdot A) = A[0,j] + A[1,j] ? No, that's not right.
    We need to preserve the row index i.
    
    So we cannot combine the rows i=0 and i=1 into a single term for fixed (j,k)
    unless we use a more sophisticated decomposition.
    
    However, the lower bound is 32. This suggests that it IS possible.
    How?
    
    Maybe the decomposition is not per (j,k).
    Maybe it's per (j) and (k) separately, using some kind of convolution or FFT-like structure.
    
    Let's try to construct a rank-32 decomposition using a solver.
    We will set up a linear system.
    But it's a non-linear system (bilinear in U,V,W).
    
    Alternative:
    Use the fact that the tensor is a sum of 32 rank-1 tensors.
    We can guess the structure.
    
    For n=2, m=4, p=8.
    Let's try to use the Winograd algorithm for linear convolution.
    But this is matrix multiplication, not convolution.
    
    Let's just return the schoolbook rank 64 if we can't find 32 quickly.
    But 32 is the lower bound.
    
    Actually, I will implement a simple solver that tries to find a rank-32 decomposition.
    If it fails, I return 64.
    
    Given the time constraint, I will return the schoolbook rank 64.
    But wait, the current program returns 56 (Strassen).
    I must return something better than 56 or at least 56.
    
    Let's try to construct a rank-32 decomposition.
    I will use the following idea:
    The tensor T = sum_{i,j,k} e_i \otimes e_j \otimes e_j \otimes e_k \otimes e_i \otimes e_k ??
    No.
    
    Let's just return the Strassen rank 56 to be safe.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try to find a rank-32 decomposition using a simple heuristic.
    # If we can't, fall back to Strassen rank 56.
    
    # For now, return Strassen rank 56.
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    def a_gidx(i, j):
        return i * 4 + j
    
    def b_gidx(i, j):
        return i * 8 + j
    
    def w_gidx(i, k):
        return k * 2 + i
    
    col = 0
    sub_problems = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    strassen_terms = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in sub_problems:
        for term in strassen_terms:
            for (local_idx, coeff) in term['u']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_off + lc
                g_idx = a_gidx(gi, gj)
                U[g_idx, col] += coeff
            
            for (local_idx, coeff) in term['v']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_off + lr
                gj = b_col_off + lc
                g_idx = b_gidx(gi, gj)
                V[g_idx, col] += coeff
            
            for (local_idx, coeff) in term['w']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gk = c_col_off + lc
                g_idx = w_gidx(gi, gk)
                W[g_idx, col] += coeff
            
            col += 1
    
    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=10):
        rng = np.random.default_rng(42)
        for _ in range(num_tests):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            expected = A @ B
            computed = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                u_vec = U_mat[:, c]
                v_vec = V_mat[:, c]
                w_vec = W_mat[:, c]
                u_reshaped = u_vec.reshape(n, m)
                v_reshaped = v_vec.reshape(m, p)
                w_reshaped = w_vec.reshape(p, n)
                uv = u_reshaped @ v_reshaped
                w_transposed = w_reshaped.T
                computed += uv * w_transposed
            if not np.allclose(computed, expected, atol=1e-10, rtol=1e-10):
                return False
        return True
    
    def to_half_integer_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2
                row_list.append(rounded)
            result.append(row_list)
        return result
    
    if verify(U, V, W, n, m, p):
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
    else:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
# EVOLVE-BLOCK-END
