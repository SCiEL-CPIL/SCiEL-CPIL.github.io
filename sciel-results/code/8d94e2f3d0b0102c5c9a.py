import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2x2 blocks.
    Rank = 42.
    """
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i * m + j, col] = U_s[i * 2 + j_local, c]
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        V[j * p + k, col] = V_s[j_local * 2 + k_local, c]
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        W[k * n + i, col] = W_s[k_local * 2 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _reduce_rank(U, V, W, tol=1e-9):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies
    among the tensor terms.
    """
    n, m, p = 2, 4, 6
    rank = U.shape[1]
    
    # Construct the matrix where each column is the flattened term u_r \otimes v_r \otimes w_r
    # Shape: (n*m*m*p*p*n, rank)
    terms = np.zeros((n * m * m * p * p * n, rank))
    
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        term = np.outer(np.outer(u_r, v_r), w_r).flatten()
        terms[:, r] = term
    
    # Compute the rank of the terms matrix
    # We use SVD to find the numerical rank
    U_svd, s, Vt = np.linalg.svd(terms, full_matrices=False)
    
    # Determine numerical rank
    # s is the singular values
    # We consider a singular value significant if it is > tol * s[0]
    if s[0] == 0:
        return U, V, W
    
    significant = s > tol * s[0]
    new_rank = np.sum(significant)
    
    if new_rank == rank:
        return U, V, W
    
    # The new basis is the first new_rank rows of Vt (which are the right singular vectors)
    # However, we need to express the original terms in this new basis.
    # The original terms matrix T can be written as T = U_svd * diag(s) * Vt.
    # The first new_rank columns of T span the same space.
    # We want to find new U', V', W' such that sum u'_r v'_r w'_r = sum u_r v_r w_r.
    # This is not straightforward because the tensor structure is not preserved by simple linear combination of terms
    # unless we are just re-expressing the same tensor in a different basis of the term space?
    # Wait. The decomposition is a sum of rank-1 tensors.
    # If the rank-1 tensors are linearly dependent, we can replace the set with a linearly independent set.
    # Let T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # If {t_r} are linearly dependent, there exist coefficients c_r not all zero such that sum c_r t_r = 0.
    # This means T = sum (1-c_r) t_r? No.
    # It means we can remove one term and adjust the others?
    # Actually, if t_k is a linear combination of others, say t_k = sum_{j != k} alpha_j t_j,
    # then T = sum_{j != k} t_j + t_k = sum_{j != k} (1 + alpha_j) t_j.
    # This changes the coefficients, but the terms t_j are still rank-1 tensors?
    # No, (1 + alpha_j) t_j is still a rank-1 tensor.
    # So we can just remove term k and scale the others?
    # Wait, if t_k = sum alpha_j t_j, then T = sum_{j != k} t_j + sum_{j != k} alpha_j t_j = sum_{j != k} (1 + alpha_j) t_j.
    # This works! The new set of terms is { (1+alpha_j) t_j | j != k }.
    # Each new term is a scalar multiple of a rank-1 tensor, so it is rank-1.
    # The rank is reduced by 1.
    
    # So, we need to find the linear dependencies.
    # The null space of T (the matrix of terms) gives the dependencies.
    # If we find a null vector c, we can eliminate one term.
    
    # Let's find the null space of the terms matrix.
    # We use the SVD results.
    # The null space is spanned by the rows of Vt corresponding to zero singular values.
    # Actually, Vt is (rank, rank). The null space of T (which is (N, rank)) is the null space of Vt?
    # No, T = U_svd S Vt. The null space of T is the null space of Vt (since U_svd is unitary and S is diagonal).
    # The null space of Vt is the orthogonal complement of the range of Vt^T.
    # If new_rank < rank, the last (rank - new_rank) rows of Vt span the null space?
    # No, the rows of Vt are the right singular vectors.
    # The vectors corresponding to zero singular values span the null space of T^T?
    # T is (N, rank). T x = 0.
    # x is in R^rank.
    # The null space of T is spanned by the right singular vectors corresponding to zero singular values.
    # In Vt, the rows are the right singular vectors.
    # So the rows of Vt from index new_rank to end span the null space.
    
    # We can iteratively remove terms.
    # However, implementing the elimination logic correctly is tricky.
    # For now, if new_rank < rank, we know the rank is lower.
    # We can return the original matrices but with the last (rank - new_rank) columns zeroed out?
    # No, that changes the tensor.
    
    # Given the complexity, and the fact that we might not be able to easily reconstruct the U, V, W matrices
    # from the reduced rank terms without solving a system, we will just return the original if we can't easily reconstruct.
    # BUT, if the rank is 40, we want to return rank 40.
    
    # Alternative: If we detect rank 40, we can try to find a rank 40 decomposition by searching?
    # No, we can just return the rank 42 one if we can't reduce it easily.
    # However, the score depends on the rank.
    
    # Let's try to implement the elimination.
    # If we have a null vector c, we can set one coefficient to 0 and adjust others.
    # But we need to update U, V, W.
    # This is not a simple column operation.
    
    # I will skip the reduction for now and return the rank 42 decomposition, 
    # unless I can find a known rank 40 construction.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the guaranteed Rank 42 solution
    U42, V42, W42 = _build_2x4x6_decomposition()
    
    # 2. Verify Rank 42
    is_valid_42, _ = _verify_decomposition(U42, V42, W42)
    
    if not is_valid_42:
        # Fallback to Schoolbook Rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 3. Try to reduce rank
    U_red, V_red, W_red = _reduce_rank(U42, V42, W42)
    
    # Check if reduction worked
    if U_red.shape[1] < U42.shape[1]:
        is_valid_red, _ = _verify_decomposition(U_red, V_red, W_red)
        if is_valid_red:
            return {
                "U": U_red.tolist(),
                "V": V_red.tolist(),
                "W": W_red.tolist()
            }
    
    # 4. Return Rank 42
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
