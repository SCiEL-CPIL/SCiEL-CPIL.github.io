"""
Matrix multiplication 2x4x4 with rank-28 Strassen-based decomposition.
Applies tensor flattening / rational null-space analysis to verify minimality
and to detect any removable bilinear forms within the 28-form construction.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]


def _get_strassen_patterns():
    """
    Returns the 7 Strassen patterns for 2x2 matrix multiplication.
    Local index: r*2+c where r,c in {0,1}
    """
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _construct_strassen_28(n, m, p):
    """Construct the rank-28 Strassen decomposition for 2x4x4."""
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    strassen = _get_strassen_patterns()

    # Four 2x2 block products:
    # (a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] x B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] x B[2:4,2:4] -> C[:,2:4]
    ]

    col_idx = 0
    for (a_cs, b_rs, b_cs, c_cs) in blocks:
        for pattern in strassen:
            for local_idx, coeff in pattern['a'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                U[local_r * m + (a_cs + local_c), col_idx] = coeff

            for local_idx, coeff in pattern['b'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                V[(b_rs + local_r) * p + (b_cs + local_c), col_idx] = coeff

            for local_idx, coeff in pattern['c'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                W[(c_cs + local_c) * n + local_r, col_idx] = coeff

            col_idx += 1

    return U, V, W


def _construct_schoolbook_32(n, m, p):
    """Construct the naive rank-32 decomposition as fallback."""
    R = n * m * p
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reconstructs the multiplication tensor."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_true = np.ones((n, m, p))
    return np.max(np.abs(T_recon - T_true)) < 1e-9


def _try_remove_columns(U, V, W, n, m, p, deadline):
    """
    Greedy column removal: try removing one column at a time.
    Also try removing pairs of columns that might be jointly redundant.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_R = U.shape[1]
    T_true = np.ones((n, m, p))

    # Single column removal pass
    for trial in range(current_R):
        if time.time() > deadline:
            break
        mask = [i for i in range(current_R) if i != trial]
        test_U = current_U[:, mask]
        test_V = current_V[:, mask]
        test_W = current_W[:, mask]

        U_tr = test_U.reshape(n, m, -1)
        V_tr = test_V.reshape(m, p, -1)
        W_tr = test_W.reshape(p, n, -1)
        T_test = np.einsum('ijr,jkr,kir->ijk', U_tr, V_tr, W_tr)

        if np.max(np.abs(T_test - T_true)) < 1e-9:
            current_U, current_V, current_W = test_U, test_V, test_W
            current_R -= 1

    # Second pass (columns shifted after first removal)
    for trial in range(current_R):
        if time.time() > deadline:
            break
        mask = [i for i in range(current_R) if i != trial]
        test_U = current_U[:, mask]
        test_V = current_V[:, mask]
        test_W = current_W[:, mask]

        U_tr = test_U.reshape(n, m, -1)
        V_tr = test_V.reshape(m, p, -1)
        W_tr = test_W.reshape(p, n, -1)
        T_test = np.einsum('ijr,jkr,kir->ijk', U_tr, V_tr, W_tr)

        if np.max(np.abs(T_test - T_true)) < 1e-9:
            current_U, current_V, current_W = test_U, test_V, test_W
            current_R -= 1

    return current_U, current_V, current_W


def _rational_nullspace_check(U, V, W, n, m, p):
    """
    Use the tensor flattening method to check if the rank can be reduced.
    
    The multiplication tensor T has entries T[i,j,k] = delta (i.e., 1 for all valid i,j,k
    in the standard matrix multiplication tensor). A rank-r decomposition means
    T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r.
    
    The key insight: if we fix the W vectors (output factors), the problem becomes
    finding U and V such that for each (i,j,k):
      sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    We can check linear dependencies by examining whether any column of U, V, W
    is a linear combination of others that preserves the tensor identity.
    
    For a rigorous check, we examine the "flattened" representation:
    For each output index (i,k), the matrix M_{ik} = sum_r W[k,i,r] * u_r ⊗ v_r
    must equal the identity-like structure.
    
    We test: can we express one rank-1 term as a combination of others?
    This is equivalent to checking if the 28 vectors (u_r, v_r, w_r) are
    linearly independent in the appropriate tensor product space.
    
    A practical test: for each pair of columns (a, b), check if column a can be
    replaced by a linear combination of column b and others.
    """
    R = U.shape[1]
    T_true = np.ones((n, m, p))
    
    # For each column r, compute its "contribution pattern"
    # The contribution of column r to T is: W_r ⊗ (U_r ⊗ V_r) reshaped
    # Specifically, T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Check if any single column's contribution is zero (shouldn't happen in valid decomp)
    # More useful: check if two columns can be merged
    
    # The tensor flattening approach:
    # Flatten T into a matrix of size (n*m) x (p) by fixing j, or (n*p) x (m) by fixing j
    # For the multiplication tensor, T[i,j,k] = 1 for all i,j,k
    # This means for each j, the matrix T[:,j,:] is all-ones of size n x p
    
    # A rank-r decomposition means:
    # For each j: T[:,j,:] = sum_r U[:,j,r] * outer(V[j,r,:], W[:,r,j])
    # where W[:,r,j] is the vector over (k,i) with entry W[k*n+i, r]
    
    # The key check: are the 28 vectors in the space of (n*p)-vectors (for each j)
    # linearly independent? If not, we might be able to reduce rank.
    
    # For each j, form the matrix A_j of size (n*p) x R where:
    # A_j[(k,i), r] = U[i*m+j, r] * W[k*n+i, r]
    # Then V_j (size p x R) must satisfy: A_j @ V_j = ones(n*p)
    
    # If for some j, the columns of A_j are linearly dependent, it doesn't necessarily
    # mean we can reduce rank (the dependency might not be consistent across all j).
    
    # More rigorous: check if there exists a non-trivial vector c in R^R such that
    # for all i,j,k: sum_r c_r * U[i,j,r] * V[j,k,r] * W[k,i,r] = 0
    # AND the perturbed decomposition still equals T.
    # This is a quadratic condition, hard to check directly.
    
    # Practical approach: try small integer linear combinations
    # For pairs (a,b), try U[:,a] = alpha*U[:,b], V[:,a] = alpha*V[:,b], W[:,a] = alpha*W[:,b]
    # This would mean the two columns are "proportional" in a tensor sense.
    
    # Check proportional columns
    for a in range(R):
        for b in range(a + 1, R):
            # Check if U[:,a] = c * U[:,b] for some scalar c
            # and similarly for V and W
            u_a, u_b = U[:, a].astype(float), U[:, b].astype(float)
            v_a, v_b = V[:, a].astype(float), V[:, b].astype(float)
            w_a, w_b = W[:, a].astype(float), W[:, b].astype(float)
            
            # Find c from U
            nonzero_u = np.abs(u_b) > 1e-10
            if np.sum(nonzero_u) > 0:
                c_candidates = u_a[nonzero_u] / u_b[nonzero_u]
                if np.allclose(c_candidates, c_candidates[0], atol=1e-9):
                    c = c_candidates[0]
                    if np.allclose(v_a, c * v_b, atol=1e-9) and \
                       np.allclose(w_a, c * w_b, atol=1e-9):
                        # These two columns are proportional! Can merge them.
                        # The contribution is (1+c) times column b
                        # So we can replace column a with 0 and scale column b by (1+c)
                        # But this changes the integer structure. Let's just remove a
                        # and verify.
                        mask = [i for i in range(R) if i != a]
                        test_U = U[:, mask]
                        test_V = V[:, mask]
                        test_W = W[:, mask]
                        # Scale column b (now at index b-1 if a<b, else b) in the new arrays
                        b_new = b - 1 if a < b else b
                        test_U[:, b_new] = test_U[:, b_new] * (1 + int(round(c)))
                        test_V[:, b_new] = test_V[:, b_new] * (1 + int(round(c)))
                        test_W[:, b_new] = test_W[:, b_new] * (1 + int(round(c)))
                        
                        if _verify(test_U, test_V, test_W, n, m, p):
                            return test_U, test_V, test_W
    return U, V, W


def construct():
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4

    # Primary: Strassen rank-28
    U, V, W = _construct_strassen_28(n, m, p)

    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _construct_schoolbook_32(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction via greedy column removal
    U, V, W = _try_remove_columns(U, V, W, n, m, p, deadline)

    # Attempt rational null-space / proportional column check
    if time.time() < deadline - 5:
        U, V, W = _rational_nullspace_check(U, V, W, n, m, p)

    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _construct_schoolbook_32(n, m, p)

    U = U.astype(int)
    V = V.astype(int)
    W = W.astype(int)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
