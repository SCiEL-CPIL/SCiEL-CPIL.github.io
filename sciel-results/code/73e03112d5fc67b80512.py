"""Program for matrix multiplication 3x3 * 3x6 using a rank-minimal decomposition search."""
import itertools
import numpy as np
import scipy.linalg
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _build_matrix_mult_tensor(n, m, p):
    """Build the matrix multiplication tensor T_{i,j,k} for C = A*B where
    A is n x m, B is m x p, C is n x p.
    """
    T = np.zeros((n*m, m*p, n*p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _als_decompose(T, rank, max_iters=500, tol=1e-6, seed=42):
    """
    Perform Alternating Least Squares (ALS) to find a rank-rank decomposition.
    T is shape (n_a, n_b, n_c).
    Returns U, V, W or None if it fails to converge to a low error.
    """
    n_a, n_b, n_c = T.shape
    rng = np.random.default_rng(seed)
    
    # Initialize U, V, W with small random values
    U = rng.normal(size=(n_a, rank))
    V = rng.normal(size=(n_b, rank))
    W = rng.normal(size=(n_c, rank))
    
    # Precompute T as a list of 2D matrices for efficiency
    # T_a is n_b x n_c
    T_slices = [T[a, :, :] for a in range(n_a)]
    
    for it in range(max_iters):
        # Update U
        # For each a, T[a] = sum_r U[a,r] (V[:,r] * W[:,r])
        # Let M_r = V[:,r] * W[:,r] (outer product)
        # T[a] = sum_r U[a,r] M_r
        # This is T[a].flatten() = U[a] @ M_flat
        # M_flat is n_b*n_c x rank
        M = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
        # M is (n_b*n_c, rank)
        # For each a, solve U[a] @ M = T[a].flatten()
        # M.T @ U[a].T = T[a].flatten().T
        # We can solve all at once if we stack T_slices
        T_flat_stack = np.array([t.flatten() for t in T_slices]) # (n_a, n_b*n_c)
        # We want U (n_a, rank) such that U @ M = T_flat_stack
        # U = T_flat_stack @ M^+
        try:
            U_new, residuals, rank_m, s = np.linalg.lstsq(M.T, T_flat_stack.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed U, W:
        # T[:, b, :] = sum_r V[b,r] (U[:,r] * W[:,r])
        # Let N_r = U[:,r] * W[:,r] (outer product, shape n_a x n_c)
        # T[:, b, :].flatten() = V[b] @ N_flat
        N = np.einsum('ar,cr->acr', U, W).reshape(-1, rank)
        T_flat_stack_b = np.array([T[:, b, :].flatten() for b in range(n_b)])
        try:
            V_new, residuals, rank_n, s = np.linalg.lstsq(N.T, T_flat_stack_b.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed U, V:
        # T[:, :, c] = sum_r W[c,r] (U[:,r] * V[:,r])
        # Let P_r = U[:,r] * V[:,r] (outer product, shape n_a x n_b)
        # T[:, :, c].flatten() = W[c] @ P_flat
        P = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
        T_flat_stack_c = np.array([T[:, :, c].flatten() for c in range(n_c)])
        try:
            W_new, residuals, rank_p, s = np.linalg.lstsq(P.T, T_flat_stack_c.T, rcond=None)
            W = W_new.T
        except:
            pass
            
        # Check convergence
        # Compute error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
        if error < tol:
            break
            
    # Final error check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
    if error < 1e-3: # Threshold for accepting a candidate for rounding
        return U, V, W, error
    return None

def _round_to_half_int(M):
    """Round matrix entries to the nearest half-integer."""
    return np.round(M * 2) / 2

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=0)

def construct():
    """Construct the rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    T = _build_matrix_mult_tensor(n, m, p)
    
    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial)) # Note: W shape is (p*n, rank) in task? 
    # Task says: W[k*n+i, r] is the output factor.
    # C is n x p. Index is i*p + k.
    # In the trivial loop: W[k*n + i][column] = 1.
    # This implies W index is k*n + i.
    # So W shape is (p*n, rank).
    # Let's re-verify the task contract: "W[k*n+i,r]".
    # Yes, W is indexed by k (0..p-1) and i (0..n-1).
    # So W has p*n rows.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
                
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    best_rank = rank_trivial
    
    # Try to find a lower rank using ALS + Rounding
    # We try ranks from best_rank - 1 down to some lower bound
    # We limit the number of trials to fit in 120s
    max_ranks_to_try = 10
    for rank in range(best_rank - 1, max(best_rank - max_ranks_to_try, 1), -1):
        found = False
        # Try a few seeds
        for seed in range(5):
            res = _als_decompose(T, rank, max_iters=200, tol=1e-5, seed=seed)
            if res is not None:
                U, V, W, err = res
                # Round to half-integers
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                
                if _verify_decomposition(U_r, V_r, W_r, T):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
        if found:
            # If we found a solution, we could try to go lower, but for now we stop early if we get a good one?
            # The loop continues to try lower ranks.
            pass
            
    return best_result
