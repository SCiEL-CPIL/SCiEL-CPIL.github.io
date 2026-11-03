"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

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

def _try_triple_elimination_fast(U, V, W, n, m, p):
    """
    Try to eliminate triples of terms simultaneously.
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
    candidates = order[:20]  # Fewer candidates for triples
    
    for idx1 in range(len(candidates)):
        if _time_left() < 15:
            break
        r1 = candidates[idx1]
        for idx2 in range(idx1 + 1, len(candidates)):
            r2 = candidates[idx2]
            for idx3 in range(idx2 + 1, len(candidates)):
                r3 = candidates[idx3]
                
                mask = np.ones(current_rank, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                mask[r3] = False
                
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if _verify_full_fast(U_test, V_test, W_test, n, m, p):
                    return U_test, V_test, W_test, current_rank - 3
    
    return current_U, current_V, current_W, current_rank

def _try_linear_combination_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by solving for W vectors given fixed U,V.
    For each candidate set of terms to remove, check if the remaining terms
    can be adjusted (specifically W) to still satisfy the tensor equations.
    """
    if _time_left() < 30:
        return None, None, None, U.shape[1]
    
    current_rank = U.shape[1]
    
    # Strategy: try removing 1-2 terms at a time.
    # For each removal, fix U,V of remaining terms and solve for W.
    # The tensor equation is: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for invalid (y,z,x): sum_r U[y,r]*V[z,r]*W[x,r] = 0
    
    # For fixed U,V, this is linear in W.
    # Unknowns: W[x, r] for x in [0, p*n), r in remaining terms.
    # Equations: valid (n*m*p) + invalid (n*m*p*n*m*p - n*m*p) ... too many.
    
    # Simplification: only enforce valid equations. Check invalid separately.
    # Valid equations: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # For a given (i,k), W[k*n+i, r] appears in m equations (one per j).
    # Coefficient of W[k*n+i, r] in equation (i,j,k) is U[i*m+j, r]*V[j*p+k, r].
    
    # So for each (i,k) pair, we have m equations for R variables (W[k*n+i, r] for r in remaining).
    # Total: n*p*(m) equations, n*p*R unknowns.
    
    # If m >= R, we might have a solution.
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(U[:, r]**2) + np.sum(V[:, r]**2) + np.sum(W[:, r]**2)
        norms.append(norm_sq)
    order = np.argsort(norms)
    
    # Try removing single terms
    for r_remove in order:
        if _time_left() < 10:
            break
        
        remaining = [r for r in range(current_rank) if r != r_remove]
        R_rem = len(remaining)
        
        # Build linear system for W
        # Unknowns: W[x, r] for x in [0, p*n), r in remaining
        # Total unknowns: p*n*R_rem
        # Equations: for each (i,j,k): sum_{r in remaining} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        A = np.zeros((n*m*p, p*n*R_rem))
        b = np.ones(n*m*p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i*m*p + j*p + k  # equation index
                    for r_idx, r in enumerate(remaining):
                        # W[k*n+i, r] has index (k*n+i)*R_rem + r_idx
                        unknown_idx = (k*n + i) * R_rem + r_idx
                        A[eq, unknown_idx] = U[i*m+j, r] * V[j*p+k, r]
        
        try:
            W_sol, res, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            continue
        
        # Check valid equations
        if np.max(np.abs(A @ W_sol - b)) > 1e-8:
            continue
        
        # Build full W
        W_new = np.zeros((p*n, R_rem))
        W_new[:] = W_sol.reshape(p*n, R_rem)
        
        U_new = U[:, remaining]
        V_new = V[:, remaining]
        
        # Full verification
        if _verify_full_fast(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, R_rem
    
    # Try removing pairs
    for idx1 in range(len(order)):
        if _time_left() < 10:
            break
        r1 = order[idx1]
        for idx2 in range(idx1+1, len(order)):
            r2 = order[idx2]
            
            remaining = [r for r in range(current_rank) if r != r1 and r != r2]
            R_rem = len(remaining)
            
            A = np.zeros((n*m*p, p*n*R_rem))
            b = np.ones(n*m*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq = i*m*p + j*p + k
                        for r_idx, r in enumerate(remaining):
                            unknown_idx = (k*n + i) * R_rem + r_idx
                            A[eq, unknown_idx] = U[i*m+j, r] * V[j*p+k, r]
            
            try:
                W_sol, res, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
            except:
                continue
            
            if np.max(np.abs(A @ W_sol - b)) > 1e-8:
                continue
            
            W_new = W_sol.reshape(p*n, R_rem)
            U_new = U[:, remaining]
            V_new = V[:, remaining]
            
            if _verify_full_fast(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, R_rem
    
    return U, V, W, current_rank

def _build_equation_system(n, m, p, R):
    """
    Build the equation system for the matrix multiplication tensor.
    Each equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) is a valid product entry, else 0.
    We flatten this into a system where each variable is a product U[i,j,r]*V[j,k,r]*W[k,i,r].
    For linear reduction: fix U,V and solve for W, or fix U,W and solve for V, etc.
    """
    pass

def _try_solve_for_W(U_fixed, V_fixed, n, m, p, W_indices_to_solve, all_indices):
    """
    Given fixed U and V columns, and a set of W columns to solve for,
    check if we can find W values that satisfy the tensor equations.
    This is bilinear in W, so we can't directly linearize.
    Instead: for each W column r, the contribution to entry (i,j,k) is U[i,j,r]*V[j,k,r]*W[k,i,r].
    If U and V are fixed, then for each r, the coefficient of W[k,i,r] is U[i,j,r]*V[j,k,r].
    So the system is linear in W!
    """
    R_fixed = U_fixed.shape[1]
    R_solve = len(W_indices_to_solve)
    
    # Build the full system: for each valid (i,j,k) entry, we need:
    # sum_{r in fixed} U[i,j,r]*V[j,k,r]*W[k,i,r] + sum_{r in solve} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # The first sum is known (call it b_known), so we need:
    # sum_{r in solve} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 - b_known = b
    
    # But W[k,i,r] appears in multiple equations (for different j).
    # For a fixed r (in solve) and fixed (k,i), W[k,i,r] is a single variable.
    # It appears in equations for all j: sum_j U[i,j,r]*V[j,k,r]*W[k,i,r] ... no wait.
    
    # Actually the tensor equation is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (or 0 for invalid)
    
    # For fixed U,V: the unknowns are W[k*n+i, r] for r in solve_indices.
    # Each equation (i,j,k) involves W[k*n+i, r] for all r in solve_indices, with coefficient U[i*m+j,r]*V[j*p+k,r].
    # But W[k*n+i, r] only appears in equations where the output index is (k,i), i.e., for all j.
    
    # So for each (k,i) pair and each r in solve, W[k*n+i, r] appears in m equations (one per j).
    # Total unknowns: p*n*R_solve
    # Total equations: n*m*p (but only n*m*p are valid; the rest must be 0)
    
    # This is a large linear system. Let's build it.
    num_unknowns = p * n * R_solve
    num_eqs = n * m * p  # all entries, valid ones = 1, invalid = 0
    
    # Actually, let's only use the valid equations first (n*m*p = 2*5*6=60 equations)
    # and check invalid ones separately.
    
    valid_eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_eqs.append((i, j, k))
    
    A = np.zeros((len(valid_eqs), num_unknowns))
    b = np.zeros(len(valid_eqs))
    
    for eq_idx, (i, j, k) in enumerate(valid_eqs):
        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For r in fixed: contributes to b (with sign)
        # For r in solve: contributes to A
        
        # b starts at 1, subtract fixed contributions
        b[eq_idx] = 1.0
        for r_fixed in range(R_fixed):
            b[eq_idx] -= U_fixed[i*m+j, r_fixed] * V_fixed[j*p+k, r_fixed] * W_fixed_val(k*n+i, r_fixed)
        
        # A: for each r in solve
        for sol_idx, r_solve in enumerate(W_indices_to_solve):
            unknown_idx = (k*n + i) * R_solve + sol_idx
            A[eq_idx, unknown_idx] = U_fixed[i*m+j, r_solve] * V_fixed[j*p+k, r_solve]
    
    # Solve
    try:
        W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
    
    # Check residuals
    if np.max(np.abs(A @ W_sol - b)) > 1e-8:
        return None
    
    # Check invalid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This is valid, skip
                pass
    
    # Check all invalid entries
    valid_set = set((i,j,k) for i in range(n) for j in range(m) for k in range(p))
    # Actually all (i,j,k) with i<n, j<m, k<p are valid for matrix mult
    # Invalid entries would be... hmm, in the tensor T_{i,j,k} = sum_r U[i,j,r]V[j,k,r]W[k,i,r]
    # The valid entries are ALL (i,j,k) since matrix mult produces all n*m*p entries.
    # Wait no - the tensor is defined such that T[i,j,k] = 1 if it's a "product" entry.
    # For matrix multiplication (n x m) * (m x p) -> (n x p):
    # T[i, j, k] should give (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor has 1 at positions corresponding to the summation structure.
    
    # Actually in the standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # And T_{i,j,k} = 1 for all i,j,k (since each output element (i,k) is sum over j)
    # No wait, that's not right either.
    
    # Let me re-read the verification:
    # S = (U * w[None, :]) @ V_T  gives S[y, z] where y=i*m+j, z=j*p+k
    # And we check S[i*m+j, j*p+k] == 1
    # And S[~valid_mask] == 0
    
    # So the tensor is: T[y, z, x] where x = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1, all other entries = 0
    
    # So for a given (i,k), the entry (i*m+j, j*p+k, k*n+i) = 1 for all j.
    # And entries like (i*m+j, j'*p+k, k'*n+i) where j'!=j or k'!=k are 0.
    
    # So the valid equations are: for each i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And invalid: for each (y,z,x) not of the form (i*m+j, j*p+k, k*n+i): sum_r U[y,r]*V[z,r]*W[x,r] = 0
    
    # This is a LOT of equations. Let's just verify the full thing after solving.
    
    # Build W_full
    W_full = np.zeros((p*n, R_fixed + R_solve))
    W_full[:, :R_fixed] = W_fixed_full  # need this passed in
    for sol_idx, r_solve in enumerate(W_indices_to_solve):
        W_full[:, R_fixed + sol_idx] = W_sol.reshape(p*n)
    
    return W_full

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
    if _time_left() > 25:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Triple elimination
    if _time_left() > 15:
        U_reduced, V_reduced, W_reduced, new_rank = _try_triple_elimination_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
    
    # Linear combination reduction (solve for W given U,V)
    if _time_left() > 30:
        U_red, V_red, W_red, new_rank = _try_linear_combination_reduction(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
