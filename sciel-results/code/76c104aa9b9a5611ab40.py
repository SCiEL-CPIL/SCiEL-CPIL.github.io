import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# Target tensor for multiplying (n x m) by (m x p) -> (n x p)
# Flattened indices:
#   A index a = i*m + j   (i in [0,n), j in [0,m))
#   B index b = j*p + k   (j in [0,m), k in [0,p))
#   C index c = k*n + i   (k in [0,p), i in [0,n))  (official trace convention)
# Tensor T[a, b, c] = 1 iff the triple (i, j, k) is consistent, i.e. always
# (for every i,j,k) the product contributes A[i,j]*B[j,k] to C[k,i].
# So T[a,b,c] = 1 for all (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i.
# There is exactly one (i,j,k) per (a,b,c) that is consistent; all others 0.
# ---------------------------------------------------------------------------

def _build_target_tensor(n, m, p):
    """Return dense target tensor T of shape (n*m, m*p, p*n) with 0/1 entries."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _schoolbook_terms(n, m, p):
    """Return list of rank-1 terms (u, v, w) as float arrays for schoolbook."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[a] = 1.0
                v[b] = 1.0
                w[c] = 1.0
                terms.append((u, v, w))
    return terms


def _tensor_from_terms(terms):
    """Compute T = sum_r outer(u_r, v_r, w_r) as a flat vector of length n*m*m*p*p*n.
    We return a 3D array for clarity but work with flat for speed."""
    # Determine shapes from first term
    u0, v0, w0 = terms[0]
    na, nb, nc = len(u0), len(v0), len(w0)
    T = np.zeros((na, nb, nc), dtype=np.float64)
    for u, v, w in terms:
        T += np.einsum('a,b,c->abc', u, v, w)
    return T


def _target_flat(n, m, p):
    """Flat target tensor vector."""
    T = _build_target_tensor(n, m, p)
    return T.ravel()


def _terms_to_matrices(terms, n, m, p):
    """Convert list of (u,v,w) to U (n*m x r), V (m*p x r), W (p*n x r)."""
    r = len(terms)
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    for col, (u, v, w) in enumerate(terms):
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w
    return U, V, W


def _check_exact(U, V, W, target_flat, tol=1e-9):
    """Check if sum_r outer(U[:,r], V[:,r], W[:,r]) == target."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T.ravel(), target_flat, atol=tol)


def _can_remove_term(terms, idx, target_flat, tol=1e-9):
    """Check if removing term idx still spans the target tensor.
    We need: target = sum_{r != idx} outer(u_r, v_r, w_r)
    i.e. target - sum_{r!=idx} ... = 0
    Equivalent: target is in span of remaining rank-1 tensors.
    """
    if len(terms) <= 1:
        return False
    remaining = [terms[r] for r in range(len(terms)) if r != idx]
    # Build matrix whose columns are vec(outer(u_r, v_r, w_r)) for r in remaining
    # and check if target is in column space.
    # For efficiency with small r, do least squares / rank check.
    na, nb, nc = terms[0][0].size, terms[0][1].size, terms[0][2].size
    # Build column vectors
    cols = []
    for u, v, w in remaining:
        cols.append(np.einsum('a,b,c->abc', u, v, w).ravel())
    if not cols:
        return np.allclose(target_flat, 0, atol=tol)
    M = np.column_stack(cols)
    # Check if target is in column space of M
    # Use SVD rank check: rank([M | target]) == rank(M)
    M_aug = np.column_stack([M, target_flat])
    rM = np.linalg.matrix_rank(M, tol=tol)
    rAug = np.linalg.matrix_rank(M_aug, tol=tol)
    return rM == rAug


def _can_merge_terms(terms, i, j, target_flat, tol=1e-9):
    """Check if terms i and j can be replaced by a single rank-1 term.
    Condition: outer(u_i, v_i, w_i) + outer(u_j, v_j, w_j) is rank-1 as a tensor,
    OR there exists (u,v,w) such that outer(u,v,w) = outer(u_i,v_i,w_i) + outer(u_j,v_j,w_j).
    
    A sum of two rank-1 tensors is rank-1 iff one is a scalar multiple of the other.
    So: check if (u_i, v_i, w_i) and (u_j, v_j, w_j) are "proportional" in the sense that
    there exist scalars alpha, beta not both zero such that the tensor sum is rank-1.
    
    More precisely: T_i + T_j is rank-1 iff there exist vectors u,v,w such that
    T_i + T_j = outer(u,v,w). This happens iff the two rank-1 tensors are linearly
    dependent in the space of rank-1 tensors, i.e., one is a scalar multiple of the other.
    
    Check: is (u_j, v_j, w_j) = c * (u_i, v_i, w_i) for some c?
    If so, then T_i + T_j = (1+c) * T_i which is rank-1 (if 1+c != 0).
    If 1+c == 0, then T_i + T_j = 0, which is rank 0, also fine.
    """
    u_i, v_i, w_i = terms[i]
    u_j, v_j, w_j = terms[j]
    
    # Check if u_i and u_j are proportional
    # Find c such that u_j ≈ c * u_i
    # Use non-zero entries of u_i
    nz_i = np.nonzero(u_i)[0]
    nz_j = np.nonzero(u_j)[0]
    
    if len(nz_i) == 0 or len(nz_j) == 0:
        # One is zero vector - the sum is just the other, which is rank-1
        return True
    
    # Check proportionality
    # c = u_j[k] / u_i[k] for any k where both non-zero
    common = set(nz_i) & set(nz_j)
    if not common:
        # No common support - can't be proportional unless both zero
        return False
    
    k = list(common)[0]
    if abs(u_i[k]) < tol:
        return False
    c = u_j[k] / u_i[k]
    
    # Verify u_j ≈ c * u_i
    if not np.allclose(u_j, c * u_i, atol=tol):
        return False
    # Verify v_j ≈ c * v_i
    if not np.allclose(v_j, c * v_i, atol=tol):
        return False
    # Verify w_j ≈ c * w_i
    if not np.allclose(w_j, c * w_i, atol=tol):
        return False
    
    # If c == -1, sum is zero (rank 0), which is fine
    # Otherwise sum is (1+c)*T_i, rank 1
    return True


def _merge_two_terms(terms, i, j):
    """Merge terms i and j into one rank-1 term. Returns the merged (u, v, w)."""
    u_i, v_i, w_i = terms[i]
    u_j, v_j, w_j = terms[j]
    
    # Find c such that term_j = c * term_i
    nz_i = np.nonzero(u_i)[0]
    nz_j = np.nonzero(u_j)[0]
    common = set(nz_i) & set(nz_j)
    k = list(common)[0]
    c = u_j[k] / u_i[k]
    
    # Merged term: (1+c) * term_i
    factor = 1.0 + c
    if abs(factor) < 1e-15:
        # Sum is zero - return zero term (will be removed)
        return (np.zeros_like(u_i), np.zeros_like(v_i), np.zeros_like(w_i))
    return (factor * u_i, factor * v_i, factor * w_i)


def _deflation_search(n, m, p, max_time=100.0):
    """Greedy deflation: start from schoolbook, try to reduce rank."""
    deadline = time.time() + max_time
    target_flat = _target_flat(n, m, p)
    terms = _schoolbook_terms(n, m, p)
    
    # First pass: try to merge proportional terms
    merged = True
    while merged and time.time() < deadline:
        merged = False
        for i in range(len(terms)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(terms)):
                if _can_merge_terms(terms, i, j, target_flat):
                    # Merge j into i
                    merged_term = _merge_two_terms(terms, i, j)
                    terms[i] = merged_term
                    del terms[j]
                    merged = True
                    break
            if merged:
                break
    
    # Second pass: try to remove redundant terms
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for idx in range(len(terms) - 1, -1, -1):
            if time.time() >= deadline:
                break
            # Skip zero terms
            u, v, w = terms[idx]
            if np.allclose(u, 0, atol=1e-15) and np.allclose(v, 0, atol=1e-15) and np.allclose(w, 0, atol=1e-15):
                del terms[idx]
                removed = True
                break
            if _can_remove_term(terms, idx, target_flat):
                del terms[idx]
                removed = True
                break
    
    # Verify final result
    if not terms:
        return None
    U, V, W = _terms_to_matrices(terms, n, m, p)
    if _check_exact(U, V, W, target_flat):
        return U, V, W
    return None


def _to_fraction_matrix(M):
    """Convert float matrix to list of lists of Fraction (integer/half-integer)."""
    result = []
    for row in M:
        frow = []
        for val in row:
            # Round to nearest half-integer
            rounded = round(val * 2) / 2.0
            frow.append(Fraction(rounded).limit_denominator(2))
        result.append(frow)
    return result


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try deflation search
    result = _deflation_search(n, m, p, max_time=min(100.0, deadline - time.time() - 5.0))
    if result is not None and time.time() < deadline - 10.0:
        U, V, W = result
        # Convert to fraction representation
        U_frac = _to_fraction_matrix(U)
        V_frac = _to_fraction_matrix(V)
        W_frac = _to_fraction_matrix(W)
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback: schoolbook
    terms = _schoolbook_terms(n, m, p)
    U, V, W = _terms_to_matrices(terms, n, m, p)
    U_frac = _to_fraction_matrix(U)
    V_frac = _to_fraction_matrix(V)
    W_frac = _to_fraction_matrix(W)
    return {"U": U_frac, "V": V_frac, "W": W_frac}
