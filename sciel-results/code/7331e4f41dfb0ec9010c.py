"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 120.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Standard Strassen 2x2x2 block
            # M1 = (A00 + A11)(B00 + B11)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # M3 = A00(B10 - B00)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M4 = A11(B01 - B11)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using vectorized numpy operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    # Compute the full tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Create a mask for valid entries
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid_mask[a, b, c] = True
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check invalid entries are zero
    invalid_mask = ~valid_mask
    if not np.allclose(T[invalid_mask], 0.0, atol=tol):
        return False
        
    return True

def _verify_skip_term(U, V, W, n, m, p, skip_term, tol=1e-9):
    """
    Verify that the decomposition remains valid if term skip_term is set to zero.
    """
    # Create copies with the skip_term column zeroed out
    U_mod = U.copy()
    V_mod = V.copy()
    W_mod = W.copy()
    
    U_mod[:, skip_term] = 0
    V_mod[:, skip_term] = 0
    W_mod[:, skip_term] = 0
    
    return _verify_full(U_mod, V_mod, W_mod, n, m, p, tol)

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminations=10):
    """
    Try to eliminate redundant terms one by one.
    Returns (U, V, W, new_rank).
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    eliminated = []
    
    # Try eliminating each term, starting with those that might be redundant
    # Sort by norm (smaller norm terms are more likely to be redundant)
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
    
    # Try terms in order of increasing norm
    order = np.argsort(norms)
    
    for r in order:
        if _time_left() < 10:
            break
        if len(eliminated) >= max_eliminations:
            break
        
        # Check if we can eliminate term r
        if _verify_skip_term(current_U, current_V, current_W, n, m, p, r):
            # Eliminate this term
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            eliminated.append(r)
            print(f"Eliminated term {r}, new rank: {current_rank}")
    
    return current_U, current_V, current_W, current_rank

def _try_linear_combination_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by finding linear dependencies among the terms.
    This is a heuristic approach using the tensor structure.
    """
    # For each pair of terms, check if one can be expressed as a linear combination
    # of others in a way that preserves the tensor.
    # This is complex, so we'll use a simpler heuristic:
    # Check if two terms have proportional U, V, W vectors.
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    removed = []
    
    for r1 in range(current_rank):
        if r1 in removed:
            continue
        if _time_left() < 20:
            break
        for r2 in range(r1+1, current_rank):
            if r2 in removed:
                continue
            
            # Check if term r2 is proportional to term r1 in all three matrices
            # U[:, r1] = alpha * U[:, r2]
            # V[:, r1] = alpha * V[:, r2]  
            # W[:, r1] = alpha * W[:, r2]
            
            u1 = current_U[:, r1]
            u2 = current_U[:, r2]
            v1 = current_V[:, r1]
            v2 = current_V[:, r2]
            w1 = current_W[:, r1]
            w2 = current_W[:, r2]
            
            # Check proportionality
            if np.allclose(u1, 0) and np.allclose(u2, 0):
                continue
            
            # Find alpha from non-zero elements
            mask = np.abs(u2) > 1e-10
            if not np.any(mask):
                continue
            
            alpha = u1[mask] / u2[mask]
            if not np.allclose(alpha, alpha[0], atol=1e-8):
                continue
            alpha = alpha[0]
            
            if np.allclose(u1, alpha * u2, atol=1e-8) and \
               np.allclose(v1, alpha * v2, atol=1e-8) and \
               np.allclose(w1, alpha * w2, atol=1e-8):
                # Term r1 is proportional to term r2
                # We can't simply remove one; we need to check if the combination
                # with other terms still works. For now, skip this complex case.
                pass
    
    return current_U, current_V, current_W, current_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try to eliminate redundant terms
    if _time_left() > 30:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
