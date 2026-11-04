import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1 (the matrix multiplication tensor)."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks (row 2 of A, columns 0-1 or 2-3)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W

def _compute_flattened_terms(U, V, W, n, m, p):
    """Compute the flattened tensor for each term."""
    rank = U.shape[1]
    terms = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        terms[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return terms

def _try_reduce_rank_global(U, V, W, n, m, p, time_budget):
    """Try to reduce rank by finding global linear dependencies among the flattened terms."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    while time.time() - start_time < time_budget:
        terms = _compute_flattened_terms(U, V, W, n, m, p)
        # Check rank of the terms matrix
        # We want to see if any column is a linear combination of the others
        # We can do this by checking if the rank of the matrix is less than current_rank
        # If so, there is a dependency.
        
        # To find the dependency, we can try to eliminate one column at a time
        # But a faster way is to find the null space of the terms matrix
        # If there is a non-trivial null space vector c, then sum c_r * term_r = 0
        # This means we can remove one term and adjust the others.
        
        # However, the null space approach gives a linear combination that sums to zero.
        # This allows us to remove one term if its coefficient is non-zero.
        
        # Let's find the null space
        try:
            # Use SVD to find the null space
            U_svd, s, Vh = np.linalg.svd(terms, full_matrices=False)
            # The rank is the number of non-zero singular values
            rank = np.sum(s > 1e-9)
            
            if rank < current_rank:
                # There is a dependency
                # The null space is spanned by the last (current_rank - rank) rows of Vh
                null_space = Vh[rank:, :]
                
                # Pick a dependency vector
                dep = null_space[0]
                
                # Find the index of the largest coefficient in dep
                max_idx = np.argmax(np.abs(dep))
                
                if np.abs(dep[max_idx]) < 1e-9:
                    continue
                    
                # We can eliminate term max_idx
                # dep_r * term_r = 0 for all r
                # term_max_idx = - sum_{r != max_idx} (dep_r / dep_max_idx) * term_r
                
                coeff = dep / dep[max_idx]
                coeff[max_idx] = 0 # We are eliminating this one
                
                # Create new U, V, W without the max_idx column
                U_new = np.delete(U, max_idx, axis=1)
                V_new = np.delete(V, max_idx, axis=1)
                W_new = np.delete(W, max_idx, axis=1)
                
                # Adjust the remaining columns
                for r in range(U_new.shape[1]):
                    if abs(coeff[r]) > 1e-9:
                        U_new[:, r] += coeff[r] * U[:, max_idx]
                        V_new[:, r] += coeff[r] * V[:, max_idx]
                        W_new[:, r] += coeff[r] * W[:, max_idx]
                
                # Verify the new decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    U, V, W = U_new, V_new, W_new
                    current_rank = U.shape[1]
                else:
                    # If verification fails, break to avoid infinite loop
                    break
            else:
                # No dependency found
                break
        except Exception as e:
            break
            
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank using global dependency check
    if time.time() < deadline - 30:
        time_budget = min(60, deadline - time.time() - 10)
        U_try, V_try, W_try = _try_reduce_rank_global(
            best_U, best_V, best_W, n, m, p, time_budget
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
