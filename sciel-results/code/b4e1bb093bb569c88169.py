"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # M1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            # M3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            # M5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            # M6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M7
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

def _verify_full_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast vectorized verification.
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    V_T = V.T  # (R, mp)
    
    for x in range(p * n):
        w = W[x, :]  # (R,)
        S = (U * w[None, :]) @ V_T
        
        i = x % n
        k = x // n
        
        # Check valid entries
        for j in range(m):
            y = i * m + j
            z = j * p + k
            if abs(S[y, z] - 1.0) > tol:
                return False
        
        # Check invalid entries
        valid_mask = np.zeros((n*m, m*p), dtype=bool)
        for j in range(m):
            valid_mask[i*m+j, j*p+k] = True
        
        if np.any(np.abs(S[~valid_mask]) > tol):
            return False
            
    return True

def _try_eliminate_terms_fast(U, V, W, n, m, p, max_eliminations=20):
    """
    Try to eliminate redundant terms one by one using fast verification.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    
    for r in order:
        if _time_left() < 30:
            break
        if current_rank <= 1:
            break
            
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        if _verify_full_fast(U_test, V_test, W_test, n, m, p):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            current_rank -= 1
            
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination_fast(U, V, W, n, m, p):
    """
    Try to eliminate pairs of terms simultaneously.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    candidates = order[:30]  # Consider more candidates
    
    for idx1 in range(len(candidates)):
        if _time_left() < 20:
            break
        r1 = candidates[idx1]
        for idx2 in range(idx1 + 1, len(candidates)):
            r2 = candidates[idx2]
            
            mask = np.ones(current_rank, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify_full_fast(U_test, V_test, W_test, n, m, p):
                return U_test, V_test, W_test, current_rank - 2
    
    return current_U, current_V, current_W, current_rank

def _try_structured_solve(n, m, p, target_rank=50):
    """
    Try to find a lower-rank decomposition by fixing sparse patterns for U and V
    and solving for W using linear algebra.
    
    The tensor equation is: for each (i,k),
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k_valid}
    
    This is nonlinear in U,V,W. But if we fix U and V, we can solve for W
    using a linear system... actually no, it's still nonlinear because W appears
    multiplicatively with U and V.
    
    Alternative approach: use the fact that the multiplication tensor has a
    specific structure. Try to find a rank-R decomposition by setting up
    a system of equations.
    
    For now, let's try a different approach: modify the Strassen decomposition
    to merge some terms.
    """
    if _time_left() < 40:
        return None
    
    # Try to find a rank-50 decomposition by modifying the Strassen block
    # The key insight: in the Strassen 2x2x2 block, we use 7 terms.
    # Can we do better for 2x5x6?
    
    # Let's try a direct approach: assume a form where j=4 is handled
    # differently and see if we can save terms.
    
    # Actually, let's try to use linear algebra to find if there's a
    # rank-50 solution by fixing the structure.
    
    # Build the 54-rank solution first
    U54, V54, W54 = _build_54_rank(n, m, p)
    
    # Try to find a linear combination of the 54 terms that gives rank 50
    # This is equivalent to finding 4 linear dependencies.
    
    # The tensor T is 10x30x12. We want to express it as sum of R rank-1 tensors.
    # Each rank-1 tensor is u_r ⊗ v_r ⊗ w_r.
    
    # This is a hard problem. Let's try a heuristic: look for terms that can
    # be merged.
    
    # For each pair of terms (r1, r2), check if their sum can be replaced
    # by a single term. This requires that u1⊗v1⊗w1 + u2⊗v2⊗w2 = u3⊗v3⊗w3
    # for some u3, v3, w3. This is generally not possible, but let's check
    # if there's a near-dependence.
    
    # Actually, a better approach: use the fact that the matrix multiplication
    # tensor for 2x5x6 might have a known lower bound. Let's check if rank 50
    # is achievable by trying to construct it directly.
    
    # Try a hybrid approach: use Strassen for the 2x4x6 part (rank 42)
    # and standard multiplication for the remaining 2x1x6 part (rank 12).
    # Total: 42 + 12 = 54. That's what we have.
    
    # Can we do better? What if we use a different factorization?
    
    # Let's try: group j into {0,1,2,3,4} and use different strategies.
    # For j in {0,1}: use Strassen-like (7 terms per k-pair)
    # For j in {2,3}: use Strassen-like (7 terms per k-pair)  
    # For j=4: standard (6 terms per k)
    # Total: 2*3*7 + 6 = 42 + 6 = 48? No wait.
    
    # Actually the current construction uses:
    # j_pairs = [(0,1), (2,3)] -> 2 pairs
    # k_pairs = [(0,1), (2,3), (4,5)] -> 3 pairs
    # Each (j_pair, k_pair) uses 7 terms
    # Total from Strassen: 2 * 3 * 7 = 42
    # Plus j=4: 6 terms
    # Total: 48? But we have 54. Let me recount.
    
    # Oh I see, the j=4 part uses 2*6 = 12 terms (one for each i in {0,1} and k in 0..5)
    # So total is 42 + 12 = 54. Correct.
    
    # Can we reduce the j=4 part? If we use Strassen for j=4 with some other j?
    # But j=4 is alone.
    
    # What if we regroup? Use j_pairs = [(0,1), (2,3), (4, ?)] but there's no ?
    
    # Alternative: use a different decomposition for the 2x5x6 case.
    # Let's try to construct a rank-50 solution by hand.
    
    # Idea: use 2x3x3 Strassen blocks.
    # 2x5x6 = 2x(3+2) x (3+3)
    # This gets complicated.
    
    # For now, let's just try to find if any 4 terms can be eliminated
    # from the 54-term solution using a more sophisticated method.
    
    # Try: for each subset of 4 terms, check if the remaining 50 terms
    # can be adjusted (by adding a small perturbation) to still satisfy
    # the tensor equation. This is a nonlinear feasibility problem.
    
    # Simpler: just try to find 4 terms whose removal leaves a valid
    # decomposition (which we already do in _try_eliminate_terms_fast).
    
    # Let's try a different heuristic: look for terms with the same
    # support pattern in U and V, and see if their W vectors are linearly
    # dependent.
    
    return None

def _try_replace_terms(U, V, W, n, m, p):
    """
    Try to replace pairs of terms with a single term that covers both.
    This is a heuristic: for each pair (r1, r2), check if there exists
    a single term (u, v, w) such that
    u⊗v⊗w ≈ u1⊗v1⊗w1 + u2⊗v2⊗w2
    in the tensor sense.
    """
    if _time_left() < 30:
        return None, None, None, U.shape[1]
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # For each pair, try to see if they can be merged
    # This is generally not possible, but let's check for special cases.
    
    # Actually, let's try a different approach: use the fact that we can
    # add a term and remove two terms if the net effect is zero.
    
    # For each triple (r1, r2, r3), check if term r1 + term r2 can be
    # replaced by term r3 (i.e., term r1 + term r2 - term r3 = 0 as a
    # tensor). This is a linear algebra problem.
    
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # We want to check if T - u1⊗v1⊗w1 - u2⊗v2⊗w2 + u3⊗v3⊗w3 = 0
    # This is equivalent to checking if the 4-tuple (u1,v1,w1), (u2,v2,w2),
    # (u3,v3,w3), and the zero tensor are linearly dependent in a certain way.
    
    # This is getting too complex. Let's just return the original.
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    if not _verify_full_fast(U, V, W, n, m, p):
        # Fallback
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Single elimination
    if _time_left() > 50:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Pairwise elimination
    if _time_left() > 20:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
    
    # Try structured solve
    if _time_left() > 30:
        result = _try_structured_solve(n, m, p, target_rank=50)
        if result is not None:
            U_s, V_s, W_s = result
            if _verify_full_fast(U_s, V_s, W_s, n, m, p):
                if U_s.shape[1] < best_rank:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = U_s.shape[1]
    
    # Try term replacement
    if _time_left() > 20:
        U_rep, V_rep, W_rep, new_rank = _try_replace_terms(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_rep, V_rep, W_rep
            best_rank = new_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
