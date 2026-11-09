"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(r):
                    u_val = U[i*m+j, c]
                    v_val = V[j*p+k, c]
                    w_val = W[k*n+i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += Fraction(int(u_val * 2), 2) * Fraction(int(v_val * 2), 2) * Fraction(int(w_val * 2), 2)
                if val != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Quick float verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _bclr_compress(U, V, W, n, m, p, max_rounds=20, deadline=None):
    """
    Apply BCLR-style compression: iteratively find linear dependencies among
    the rank-1 components and eliminate them to reduce rank.
    
    The key idea: if a set of rank-1 components sum to zero (or can be combined
    to cancel), we can remove them. More practically, we look for columns in
    U, V, W that are linearly dependent in a way that allows rank reduction.
    
    A simpler approach: try to find a linear transformation of the rank-1 terms
    that introduces zeros, then drop zero terms.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]
    
    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to find a column c where we can express U[:,c] as a linear combo of other columns
        # such that the corresponding V and W columns also align, allowing elimination.
        
        # Simpler heuristic: look for pairs of columns that can be combined to create zeros
        # in one of the matrices.
        
        # Try: for each pair (a, b), check if there exists alpha such that
        # alpha * U[:,a] + U[:,b] has more zeros, and similarly for V and W.
        # This is expensive, so use a simpler approach.
        
        # Alternative: use SVD-like approach on the tensor.
        # For now, try a simple approach: look for columns that are identical or negatives.
        
        # Check for duplicate or opposite columns
        for c1 in range(r):
            if c1 >= r:
                break
            for c2 in range(c1 + 1, r):
                if c2 >= r:
                    break
                # Check if columns are identical
                if np.allclose(U[:, c1], U[:, c2]) and np.allclose(V[:, c1], V[:, c2]) and np.allclose(W[:, c1], W[:, c2]):
                    # They contribute 2x the same term. We can't just remove one,
                    # but we could combine them. Actually, this doesn't reduce rank directly.
                    pass
                # Check if they are negatives
                if np.allclose(U[:, c1], -U[:, c2]) and np.allclose(V[:, c1], -V[:, c2]) and np.allclose(W[:, c1], W[:, c2]):
                    # They cancel out! Remove both.
                    keep = [c for c in range(r) if c != c1 and c != c2]
                    U = U[:, keep]
                    V = V[:, keep]
                    W = W[:, keep]
                    r -= 2
                    break
            if c1 >= r:
                break
            if r == 0:
                break
        else:
            # No simple cancellation found; break to avoid infinite loop
            # In a real BCLR, we'd do more sophisticated transformations
            break
        
        if r == 0:
            break
    
    return U, V, W, r

def _try_fixed_uv_solve_w(U, V, n, m, p, r):
    """Given fixed U and V, try to solve for W using least squares."""
    W = np.zeros((p * n, r))
    
    for i in range(n):
        for k in range(p):
            A_mat = np.zeros((m, r))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            b = np.ones(m)
            try:
                w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                return None
            
            w_scaled = w_vec * 2
            w_rounded = np.round(w_scaled)
            if np.linalg.norm(A_mat @ (w_rounded / 2) - b) > 1e-8:
                return None
            
            W[k * n + i, :] = w_rounded / 2
    
    if _verify_float(U, V, W, n, m, p):
        return W
    return None

def _random_search(n, m, p, r, max_attempts, seed_base, deadline):
    """Random search for rank-r decomposition."""
    np.random.seed(seed_base)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        scale = 1.0
        U = np.random.randint(-1, 2, size=(n * m, r)) * scale
        V = np.random.randint(-1, 2, size=(m * p, r)) * scale
        
        U_nonzero = np.any(U != 0, axis=1)
        V_nonzero = np.any(V != 0, axis=1)
        if not np.all(U_nonzero) or not np.all(V_nonzero):
            continue
        
        W = _try_fixed_uv_solve_w(U, V, n, m, p, r)
        if W is not None:
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _structured_search(n, m, p, r, deadline):
    """Try structured patterns for U and V."""
    results = []
    
    res = _random_search(n, m, p, r, 500, 0, deadline)
    if res:
        return res
    
    np.random.seed(1000)
    for attempt in range(200):
        if time.time() > deadline:
            break
        U = np.random.randint(-2, 3, size=(n * m, r))
        V = np.random.randint(-2, 3, size=(m * p, r))
        U_nonzero = np.any(U != 0, axis=1)
        V_nonzero = np.any(V != 0, axis=1)
        if not np.all(U_nonzero) or not np.all(V_nonzero):
            continue
        W = _try_fixed_uv_solve_w(U, V, n, m, p, r)
        if W is not None:
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _try_rank_reduction_via_linear_combos(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linear combinations of rank-1 terms
    that preserve the tensor but introduce more zeros, enabling further reduction.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]
    
    # Try random linear transformations of the column space
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random invertible matrix T of size r x r
        T = np.random.randn(r, r)
        try:
            if np.abs(np.linalg.det(T)) < 1e-10:
                continue
        except np.linalg.LinAlgError:
            continue
        
        # Apply transformation: new_U = U @ T, new_V = V @ T^{-1}, new_W = W @ T^{-1}
        # This preserves the tensor: sum_c U[:,c] T[c,d] * V[:,c] T^{-1}[c,d] * W[:,c] T^{-1}[c,d]
        # Wait, this isn't right. The tensor is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # If we do U' = U @ T, V' = V @ T^{-1}, W' = W @ T^{-1}, then
        # sum_c U'[i,j,c] V'[j,k,c] W'[k,i,c] = sum_c,d,e U[i,j,d] T[d,c] V[j,k,e] T^{-1}[e,c] W[k,i,f] T^{-1}[f,c]
        # This doesn't simplify nicely. 
        # 
        # Actually, the correct transformation for tensor rank preservation is:
        # U' = U @ A, V' = V @ A^{-1}, W' = W @ A^{-1} does NOT preserve the tensor.
        # 
        # The tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # If we replace c with a linear combination, we need:
        # T[i,j,k] = sum_c (sum_d U[i,j,d] A[d,c]) (sum_e V[j,k,e] B[e,c]) (sum_f W[k,i,f] C[f,c])
        # This is not generally equal to the original unless A, B, C satisfy specific conditions.
        #
        # The correct rank-preserving transformation is:
        # U' = U @ A, V' = V @ A^{-1}, W' = W @ A^{-1} only works if the tensor is
        # decomposed as sum_c U_c ⊗ V_c ⊗ W_c and we apply A to the coefficient space.
        # But here U, V, W are the coefficient matrices, so the transformation is:
        # new_U[:,c] = sum_d U[:,d] A[d,c]
        # new_V[:,c] = sum_d V[:,d] A^{-1}[d,c]  -- this is wrong
        #
        # Actually, for a tensor decomposition T = sum_c u_c ⊗ v_c ⊗ w_c,
        # if we let (u'_c, v'_c, w'_c) = sum_d (u_d, v_d, w_d) * M[d,c] for some matrix M,
        # then T = sum_c u'_c ⊗ v'_c ⊗ w'_c only if M is a permutation or something special.
        #
        # The standard rank-preserving transformation is:
        # u'_c = sum_d u_d * A[d,c]
        # v'_c = sum_d v_d * A^{-1}[c,d]  (note the index swap)
        # w'_c = w_c (unchanged) -- no, this doesn't work either.
        #
        # Let me think again. The tensor is T = sum_c u_c ⊗ v_c ⊗ w_c.
        # If we define new terms u'_c = sum_d A[d,c] u_d, then
        # sum_c u'_c ⊗ v_c ⊗ w_c = sum_c,d A[d,c] u_d ⊗ v_c ⊗ w_c
        # This is not equal to T in general.
        #
        # The correct approach: if we have a linear dependence among the terms,
        # i.e., sum_c alpha_c * u_c ⊗ v_c ⊗ w_c = 0, then we can use this to
        # reduce the rank. But finding such dependencies is the hard part.
        #
        # For now, let's just try the simple cancellation approach from _bclr_compress.
        pass
    
    return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin (evaluator kills at 150s)
    
    # Trivial solution fallback
    U_triv, V_triv, W_triv = _trivial_solution(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound for rank
    lb = max(n * m, m * p, p * n)  # = 35 for 3x5x7
    
    # Try ranks from lower bound upwards
    for r in range(lb, min(lb + 30, best_rank)):
        if time.time() > deadline:
            break
        
        rank_deadline = min(deadline, time.time() + 10)
        
        res = _structured_search(n, m, p, r, rank_deadline)
        if res is not None:
            U, V, W = res
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # Try some known good ranks
    if best_rank > 60:
        for r in [40, 45, 50, 55, 60]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                continue
            
            rank_deadline = min(deadline, time.time() + 8)
            res = _random_search(n, m, p, r, 300, r * 100, rank_deadline)
            if res is not None:
                U, V, W = res
                best_rank = r
                best_U, best_V, best_W = U, V, W
    
    # Try BCLR compression on the best solution found
    if time.time() < deadline - 10:
        U_comp, V_comp, W_comp, r_comp = _bclr_compress(
            best_U, best_V, best_W, n, m, p, 
            max_rounds=20, deadline=deadline - 5
        )
        if r_comp < best_rank:
            if _verify_float(U_comp, V_comp, W_comp, n, m, p):
                if _verify_exact(U_comp, V_comp, W_comp, n, m, p):
                    best_rank = r_comp
                    best_U, best_V, best_W = U_comp, V_comp, W_comp
    
    # Final attempt: try to improve with more random search at the best rank found
    if best_rank > lb and time.time() < deadline - 5:
        r_try = best_rank - 1
        if r_try >= lb:
            rank_deadline = min(deadline, time.time() + 5)
            res = _random_search(n, m, p, r_try, 100, best_rank * 1000, rank_deadline)
            if res is not None:
                U, V, W = res
                best_rank = r_try
                best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
