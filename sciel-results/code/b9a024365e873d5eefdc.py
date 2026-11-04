# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random
import itertools

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor.
    T[a, b, c] where a indexes (i,j) for A, b indexes (k,l) for B, c indexes (r = l*n+i) for C
    T[a,b,c] = 1 if j==k else 0
    """
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == k:
                        a = i * m + j
                        b = k * m + l
                        c = l * n + i
                        tensor[a, b, c] = 1
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    recon = np.zeros_like(tensor, dtype=np.float64)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.max(np.abs(recon - tensor)) < tol

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_linear_dependency(vectors):
    """
    Find a linear dependency among the given list of numpy arrays (vectors).
    Returns a list of coefficients c such that sum c_i * v_i = 0, or None if independent.
    Uses SVD for numerical stability.
    """
    # Stack vectors into a matrix (rows are vectors)
    M = np.vstack(vectors)
    # Compute SVD
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Find the rank
    tol = 1e-9
    rank = np.sum(s > tol)
    if rank < len(vectors):
        # The null space is spanned by the last columns of Vt
        # Vt has shape (N, N) if M is (N, D) and we use full_matrices=True?
        # No, SVD of M (NxD) gives Vt of shape (D, D) if full_matrices=True.
        # We need the null space of M.
        # The null space is the orthogonal complement of the row space.
        # The row space is spanned by the first 'rank' rows of U? No, columns of U.
        # The null space of M is spanned by the last (N - rank) rows of Vt?
        # Wait, M x = 0.
        # SVD: M = U S Vt.
        # M x = U S Vt x = 0.
        # Let y = Vt x. Then U S y = 0.
        # Since U is orthogonal, S y = 0.
        # S is diagonal with 'rank' non-zero entries.
        # So y must have 0 in the first 'rank' components.
        # The last (N - rank) components of y can be anything.
        # x = V y.
        # So the null space is spanned by the last (N - rank) columns of V.
        # V is the conjugate transpose of Vt.
        # So we need the last (N - rank) rows of Vt? No, columns of V.
        # V = Vt.T.
        # So we take the last (N - rank) columns of V, which are the last (N - rank) rows of Vt.T?
        # Let's just compute the null space directly using the SVD components.
        # Vt shape is (D, D). The null space of M (NxD) is a subspace of R^D?
        # No, we have N vectors in R^D. We are looking for coefficients c in R^N such that M c = 0.
        # This is the null space of M^T? No.
        # M is N x D. We want c (D x 1)? No, c is N x 1.
        # We want sum c_i v_i = 0.
        # v_i are rows of M.
        # So we want M^T c = 0? No.
        # If v_i are rows, then M c = 0 is not right.
        # Let V be the matrix with v_i as rows.
        # We want c such that V c = 0? No, c is a vector of coefficients.
        # The linear combination is sum c_i v_i.
        # This is equivalent to c^T V = 0 (if v_i are rows).
        # Or V^T c = 0.
        # So we need the null space of V^T.
        # V^T is D x N.
        # The null space of V^T is in R^N.
        # SVD of V^T: V^T = U S Vt.
        # Null space is spanned by the last (N - rank) columns of Vt.
        
        # Let's just use the null space of M (N x D) is not what we want.
        # We want the null space of the matrix formed by the vectors as columns?
        # If we form a matrix A where columns are v_i (D x N), then A c = 0.
        # This is the null space of A.
        
        A = np.vstack(vectors).T # D x N
        U_A, s_A, Vt_A = np.linalg.svd(A, full_matrices=True)
        # Vt_A is N x N.
        # The null space is spanned by the last (N - rank) rows of Vt_A.
        null_space = Vt_A[rank:, :]
        if null_space.shape[0] > 0:
            # Pick the first vector in the null space
            return null_space[0]
    return None

def _try_rank_reduction(n, m, p, start_rank, deadline):
    """Try to reduce rank from start_rank by finding linear dependencies."""
    tensor = _build_tensor(n, m, p)
    best_rank = start_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Get the initial rank-1 tensors from schoolbook
    U_init = np.array(best_result["U"])
    V_init = np.array(best_result["V"])
    W_init = np.array(best_result["W"])
    
    current_rank = start_rank
    current_U = U_init
    current_V = V_init
    current_W = W_init
    
    # We will iteratively try to reduce the rank by 1
    # by finding a linear dependency among the current rank-1 terms.
    
    for _ in range(10): # Max 10 reductions
        if time.time() > deadline:
            break
            
        if current_rank <= 42: # Lower bound
            break
            
        # Flatten the current rank-1 tensors
        vectors = []
        for r in range(current_rank):
            vec = np.einsum('a,b,c->abc', current_U[:, r], current_V[:, r], current_W[:, r]).flatten()
            vectors.append(vec)
            
        # Find a linear dependency
        c = _find_linear_dependency(vectors)
        if c is None:
            break
            
        # Identify the index of the term to eliminate (largest coefficient)
        idx_to_eliminate = np.argmax(np.abs(c))
        c_k = c[idx_to_eliminate]
        
        if abs(c_k) < 1e-9:
            continue
            
        # Update coefficients for the remaining terms
        # T_total = sum_{r} T_r
        # T_k = - sum_{r != k} (c_r / c_k) T_r
        # T_total = sum_{r != k} T_r + T_k = sum_{r != k} (1 - c_r/c_k) T_r
        
        new_U = []
        new_V = []
        new_W = []
        
        for r in range(current_rank):
            if r == idx_to_eliminate:
                continue
            factor = 1 - c[r] / c_k
            # The factor is a scalar. We can multiply U[:, r] by it.
            # To keep entries as fractions/floats, we just store the scaled vectors.
            # However, the interface expects U, V, W matrices.
            # We can absorb the factor into one of the matrices, e.g., U.
            new_U.append(current_U[:, r] * factor)
            new_V.append(current_V[:, r])
            new_W.append(current_W[:, r])
            
        if not new_U:
            break
            
        current_U = np.vstack(new_U).T if len(new_U) > 1 else np.array(new_U[0]).reshape(-1, 1)
        # Wait, np.vstack(new_U) gives (N, Rank). We want (N, Rank) for U.
        # new_U is a list of arrays of shape (N,).
        # np.array(new_U).T gives (N, Rank).
        current_U = np.array(new_U).T
        current_V = np.array(new_V).T
        current_W = np.array(new_W).T
        
        current_rank -= 1
        
        # Verify the new decomposition
        # Convert to lists for the result format
        U_list = [[float(x) for x in row] for row in current_U.T]
        V_list = [[float(x) for x in row] for row in current_V.T]
        W_list = [[float(x) for x in row] for row in current_W.T]
        
        if _verify_decomposition(U_list, V_list, W_list, tensor):
            best_rank = current_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
        else:
            # If verification fails, we might have numerical issues or logic error.
            # Break to return the last valid one.
            break
            
    return best_rank, best_result

def _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=100):
    """Random search for a rank-r decomposition with integer/half-integer entries."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Values to try: integers in a small range
    values = [-2, -1, 0, 1, 2]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W with small integer entries
        U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n_m)]
        V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m_p)]
        W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p_n)]
        
        # Convert to float for verification
        U_f = np.array(U)
        V_f = np.array(V)
        W_f = np.array(W)
        
        recon = np.zeros((n_m, m_p, p_n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
        
        if np.max(np.abs(recon - tensor)) < 1e-9:
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try to reduce rank using linear dependency search
    if time.time() < deadline - 10:
        new_rank, new_result = _try_rank_reduction(n, m, p, best_rank, deadline - 10)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
            
    # If linear dependency didn't reduce much, try random search on the result
    if time.time() < deadline - 5:
        # Try to reduce by 1 more from the current best
        target_rank = best_rank - 1
        found = _random_search_for_rank(n, m, p, target_rank, _build_tensor(n, m, p), deadline - 5, max_iters=50)
        if found is not None:
            best_rank = target_rank
            best_result = found

    return best_result

# EVOLVE-BLOCK-END
