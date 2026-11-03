"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    deadline = time.time() + 130  # safety margin
    
    # Try to find a lower-rank decomposition
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try rank from 25 upward
    for R in range(25, rank_schoolbook):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            rng = np.random.default_rng(attempt * 1000 + R)
            U = rng.standard_normal((nm, R))
            V = rng.standard_normal((mp, R))
            W = rng.standard_normal((pn, R))
            
            # Simple iterative refinement
            for it in range(50):
                if time.time() > deadline:
                    break
                # Update U
                for r in range(R):
                    # T[:,:,r] approx = U[:,r] outer V[:,r] outer W[:,r]
                    # For fixed r, the contribution to T is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                    # We want to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||
                    # Gradient descent on U[:,r]
                    # This is complex; use a simpler update
                    pass
                # This approach is too slow. Let's try something else.
                break
        
        # Alternative: use a greedy approach
        # Try to build a decomposition greedily
        U_g = np.zeros((nm, R))
        V_g = np.zeros((mp, R))
        W_g = np.zeros((pn, R))
        residual = T.copy()
        
        for r in range(R):
            if time.time() > deadline:
                break
            # Find best rank-1 approximation to residual
            # This is a 3-way tensor rank-1 approximation, which is NP-hard in general
            # Use a simple heuristic: random search
            best_err = np.inf
            best_u, best_v, best_w = None, None, None
            for trial in range(100):
                u = rng.standard_normal(nm)
                v = rng.standard_normal(mp)
                w = rng.standard_normal(pn)
                term = np.einsum('i,j,k->ijk', u, v, w)
                err = np.linalg.norm(residual - term)
                if err < best_err:
                    best_err = err
                    best_u, best_v, best_w = u, v, w
            if best_u is not None:
                U_g[:, r] = best_u
                V_g[:, r] = best_v
                W_g[:, r] = best_w
                residual -= np.einsum('i,j,k->ijk', best_u, best_v, best_w)
        
        # Check if residual is close to zero
        if np.linalg.norm(residual) < 1e-8:
            best_rank = R
            best_U, best_V, best_W = U_g, V_g, W_g
            break
    
    if best_U is not None:
        # Round to nearest half-integer
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
