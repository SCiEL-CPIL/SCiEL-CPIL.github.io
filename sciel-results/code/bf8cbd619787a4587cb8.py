"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p

    # Build the target tensor as a 3D array T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook solution as fallback
    def schoolbook():
        U = np.zeros((nm, schoolbook_rank))
        V = np.zeros((mp, schoolbook_rank))
        W = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = schoolbook_rank

    # Try to find lower rank decompositions
    # For a given rank r, we need U (nm x r), V (mp x r), W (pn x r)
    # such that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    # This is a system of nm*mp*pn equations in nm*r + mp*r + pn*r unknowns
    # But it's bilinear/multilinear, so we use iterative approach

    # Use random restarts with alternating least squares
    rng = np.random.default_rng(42)
    
    def try_rank(r, max_iters=200, n_restarts=5):
        """Try to find rank-r decomposition using ALS."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = rng.standard_normal((nm, r))
            V = rng.standard_normal((mp, r))
            W = rng.standard_normal((pn, r))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each a, U[a,:] = argmin sum_{b,c} (T[a,b,c] - sum_c' U[a,c']*V[b,c']*W[c,c'])^2
                # This is: U[a,:] = (V^T ⊙ W^T flattened)^{-1} * T[a,:,:].flatten()
                # More precisely: U[a,:] solves (V^T ⊗ W^T) * U[a,:] = T[a,:,:].flatten()
                # where ⊗ is Kronecker product
                
                # Precompute: for each a, we need to solve (W ⊙ V)^T * U[a,:] = T[a,:,:].reshape(-1)
                # Actually: sum_c U[a,c] * V[b,c] * W[c_index,c] = T[a,b,c_index]
                # Fix a: U[a,:] is unknown. The system is:
                # For each (b, c_index): sum_c U[a,c] * V[b,c] * W[c_index,c] = T[a,b,c_index]
                # This is: (V^T ⊙ W^T) * U[a,:] = T[a,:,:].flatten()
                # where V^T ⊙ W^T is (mp*pn) x r matrix with entry [(b,c_index), c] = V[b,c]*W[c_index,c]
                
                # Build the system matrix once (it doesn't depend on a)
                # M_U[a] would be the same for all a: (V^T ⊙ W^T)
                M_U = (V.T * W.T).reshape(r, mp * pn).T  # (mp*pn, r)
                # Actually: M_U[(b, c_idx), c] = V[b, c] * W[c_idx, c]
                M_U = (V[:, :, None] * W[None, :, :]).transpose(1, 2, 0).reshape(mp * pn, r)
                
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                    except:
                        U[a, :] = rng.standard_normal(r)
                
                # Update V
                M_V = (U[:, :, None] * W[None, :, :]).transpose(1, 2, 0).reshape(nm * pn, r)
                for b in range(mp):
                    # T[:, b, :] has shape (nm, pn)
                    rhs = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                    except:
                        V[b, :] = rng.standard_normal(r)
                
                # Update W
                M_W = (U[:, :, None] * V[None, :, :]).transpose(1, 2, 0).reshape(nm * mp, r)
                for c_idx in range(pn):
                    rhs = T[:, :, c_idx].flatten()
                    try:
                        W[c_idx, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                    except:
                        W[c_idx, :] = rng.standard_normal(r)
                
                # Check convergence
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(recon - T) / np.linalg.norm(T)
                if err < 1e-10:
                    return U, V, W
            
            # Final check
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(recon - T) / np.linalg.norm(T)
            if err < 1e-10:
                return U, V, W
        
        return None

    # Try ranks from schoolbook_rank-1 downward
    for r in range(schoolbook_rank - 1, max(1, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=100, n_restarts=3)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break  # Found the lowest rank, stop

    # Convert to lists for output
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
