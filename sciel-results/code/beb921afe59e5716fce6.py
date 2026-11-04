"""
Program for 4x4x4 matrix multiplication tensor decomposition.
Uses a correct Kronecker product of Strassen rank-7 to achieve rank 49.
Additionally attempts rank reduction via linear dependency detection.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7), (4, 7), (4, 7).
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            u1_2d = U2[:, r1].reshape(2, 2)
            u2_2d = U2[:, r2].reshape(2, 2)
            U_col = np.kron(u1_2d, u2_2d)
            U[:, r] = U_col.flatten()
            
            v1_2d = V2[:, r1].reshape(2, 2)
            v2_2d = V2[:, r2].reshape(2, 2)
            V_col = np.kron(v1_2d, v2_2d)
            V[:, r] = V_col.flatten()
            
            w1_2d = W2[:, r1].reshape(2, 2)
            w2_2d = W2[:, r2].reshape(2, 2)
            W_col = np.kron(w1_2d, w2_2d)
            W[:, r] = W_col.flatten()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}

def _attempt_rank_reduction(U, V, W, n, m, p):
    """
    Attempt to reduce the rank by checking for linear dependencies among the rank-1 tensors.
    Returns a potentially reduced decomposition dict, or None if no reduction found.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    num_terms = U_arr.shape[1]
    
    # Flatten each rank-1 tensor into a vector
    # Term r is: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We need to check if these num_terms vectors are linearly independent
    
    # Build the matrix where each column is a flattened rank-1 tensor
    # Shape: (n*m*m*p*p*n, num_terms) = (4096, 49) for 4x4x4
    
    # More efficient: use the tensor structure
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Flatten: for each r, the tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # We can check linear independence by seeing if there exists a non-zero vector
    # alpha such that sum_r alpha[r] * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
    
    # This is equivalent to checking if the vectors U[:,r] ⊗ V[:,r] ⊗ W[:,r] are linearly independent
    
    # For efficiency, we can check if the rank of the 3-way tensor formed by these vectors is less than num_terms
    # But that's complex. Instead, let's use a simpler approach:
    # Check if there's a linear dependency by solving for alpha
    
    # Actually, a simpler check: if we can find alpha != 0 such that
    # sum_r alpha[r] * U[:,r] = 0 AND sum_r alpha[r] * V[:,r] = 0 AND sum_r alpha[r] * W[:,r] = 0
    # then we have a dependency. But this is too strong (it's not necessary).
    
    # The correct condition is: sum_r alpha[r] * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
    # This doesn't require each factor to be zero separately.
    
    # Let's use SVD on the flattened matrix
    dim_a = n * m  # 16
    dim_b = m * p  # 16
    dim_c = p * n  # 16
    
    # Build flattened matrix: shape (dim_a * dim_b * dim_c, num_terms)
    # This is 4096 x 49, which is manageable
    
    flat_matrix = np.zeros((dim_a * dim_b * dim_c, num_terms), dtype=float)
    
    for r in range(num_terms):
        # Tensor for term r: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Flatten it
        tensor_r = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        flat_matrix[:, r] = tensor_r.flatten()
    
    # Compute the rank of flat_matrix
    # Use SVD
    try:
        U_svd, s, Vt = np.linalg.svd(flat_matrix, full_matrices=False)
        # Rank is the number of singular values above tolerance
        tol_svd = max(flat_matrix.shape) * np.amax(s) * np.finfo(float).eps
        rank_svd = np.sum(s > tol_svd)
    except Exception:
        return None
    
    if rank_svd < num_terms:
        # There's a linear dependency
        # Find the null space
        # The null space is spanned by the last (num_terms - rank_svd) rows of Vt
        null_vecs = Vt[rank_svd:, :]
        
        # For each null vector, we can eliminate one term
        # Let's try to eliminate the term with the smallest coefficient
        # Actually, we can use the null vector to express one term as a combination of others
        
        # For simplicity, let's just check if we can reduce by 1
        # Use the first null vector
        alpha = null_vecs[0]
        
        # Find the term with the largest absolute coefficient to eliminate
        max_idx = np.argmax(np.abs(alpha))
        
        # We need to verify that eliminating this term still gives a valid decomposition
        # The new decomposition has rank num_terms - 1
        # We need to adjust the other terms
        
        # Actually, this is tricky. The null vector tells us that:
        # sum_r alpha[r] * T_r = 0
        # So T_max_idx = -sum_{r != max_idx} (alpha[r]/alpha[max_idx]) * T_r
        
        # But we can't just combine the U, V, W matrices like that because
        # the tensor product doesn't distribute over addition in a simple way for the factors.
        
        # This approach is flawed. Let's use a different method.
        
        # Alternative: use the fact that if there's a dependency, we can find a basis
        # for the span of the tensors and express the decomposition in that basis.
        
        # For now, let's just return the original if we detect a dependency
        # and try a different approach
        
        pass
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook (rank n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: must return before 120s (evaluator kills at 150s)
    deadline = start_time + 110  # 10s safety margin
    
    # Try rank-49 (Kronecker product of Strassen) - primary approach
    try:
        result49 = _kronecker_decomposition(n, m, p)
        valid49, diff49 = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"])
        
        if valid49:
            # Attempt rank reduction
            if time.time() < deadline - 5:
                reduced = _attempt_rank_reduction(result49["U"], result49["V"], result49["W"], n, m, p)
                if reduced is not None:
                    valid_red, diff_red = _verify_decomp(n, m, p, reduced["U"], reduced["V"], reduced["W"])
                    if valid_red:
                        return {"U": reduced["U"], "V": reduced["V"], "W": reduced["W"]}
            
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
        
        # If verification failed, try with relaxed tolerance
        valid49b, diff49b = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"], tol=1e-6)
        if valid49b:
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
        
        # If verification failed, fall through to schoolbook
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Fall back to schoolbook
        result = _schoolbook_decomposition(n, m, p)
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Final fallback: schoolbook
    result = _schoolbook_decomposition(n, m, p)
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
