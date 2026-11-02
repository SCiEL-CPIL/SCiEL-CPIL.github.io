"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition."""

import time
import numpy as np

DEADLINE_S = 110.0
DIMENSIONS = [4, 5, 5]


def _validate_decomposition(U, V, W, n, m, p, tol=1e-6):
    R = U.shape[1]
    T = np.ones((n, m, p))
    for r in range(R):
        u = np.zeros((n, m))
        v = np.zeros((m, p))
        w = np.zeros((p, n))
        for i in range(n):
            for j in range(m):
                u[i, j] = U[i*m+j, r]
        for j in range(m):
            for k in range(p):
                v[j, k] = V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                w[k, i] = W[k*n+i, r]
        # outer product contribution
        contrib = np.einsum('ij,jk,ki->ijk', u, v, w)
        T = T - contrib
    return np.allclose(T, 0, atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _try_decompose(n, m, p, target_rank, max_iters=200, seed=42):
    """Try to find a rank-target_rank decomposition using iterative peeling."""
    t_start = time.time()
    rng = np.random.RandomState(seed)
    
    T = np.ones((n, m, p))
    
    for restart in range(max_iters):
        if time.time() - t_start > DEADLINE_S - 25:
            break
        
        residual = T.copy()
        U_cols, V_cols, W_cols = [], [], []
        success = True
        
        for r in range(target_rank):
            if time.time() - t_start > DEADLINE_S - 25:
                success = False
                break
            
            # Try multiple random attempts to peel a rank-1 component
            best_peel = None
            best_score = np.inf
            
            for attempt in range(80):
                # Random u in R^n, v in R^m
                u = rng.randn(n)
                v = rng.randn(m)
                
                # Compute optimal w: minimize ||residual - outer(u,v,w)||_F
                # w[k] = sum_{i,j} residual[i,j,k] * u[i] * v[j] / (||u||^2 * ||v||^2)
                denom = np.sum(u**2) * np.sum(v**2)
                if abs(denom) < 1e-10:
                    continue
                
                # numerator[k] = sum_{i,j} residual[i,j,k] * u[i] * v[j]
                # = sum_i u[i] * sum_j residual[i,j,k] * v[j]
                A = np.tensordot(residual, v, axes=([1], [0]))  # shape (n, p)
                numerator = np.dot(u, A)
                w = numerator / denom
                
                # Compute the peeled tensor
                peeled = np.einsum('i,j,k->ijk', u, v, w)
                new_residual = residual - peeled
                
                # Score: how well does this peel? Use residual norm
                # Also check if the peel is "meaningful" (not too small)
                peel_norm = np.sum(np.abs(peeled))
                res_norm = np.sum(np.abs(new_residual))
                
                # We want peel to be significant and residual to decrease
                if peel_norm < 1e-6:
                    continue
                score = res_norm / peel_norm
                if score < best_score:
                    best_score = score
                    best_peel = (u, v, w, new_residual)
            
            if best_peel is None:
                success = False
                break
            
            u, v, w, new_residual = best_peel
            residual = new_residual
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        if not success:
            continue
        
        # Check if residual is close to zero
        if np.sum(np.abs(residual)) < 1e-6:
            # Build full U, V, W
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                u_r, v_r, w_r = U_cols[r], V_cols[r], W_cols[r]
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = u_r[i]
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, r] = v_r[j]
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, r] = w_r[k]
            
            if _validate_decomposition(U, V, W, n, m, p, tol=1e-5):
                return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Try decreasing ranks
    # For 4x5x5, try from 95 down to maybe 35
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() - t0 > DEADLINE_S - 30:
            break
        
        result = _try_decompose(n, m, p, target_rank, max_iters=30, seed=target_rank)
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_result = (U_new, V_new, W_new)
            # Don't break - try to go lower
            # But if we found a good one, continue trying lower
    
    # Return the best result
    def _to_list(M):
        out = []
        for row in M:
            lrow = []
            for x in row:
                xv = float(x)
                if abs(xv - round(xv)) < 1e-8:
                    lrow.append(int(round(xv)))
                else:
                    lrow.append(xv)
            out.append(lrow)
        return out
    
    U_best, V_best, W_best = best_result
    return {
        "U": _to_list(np.array(U_best)),
        "V": _to_list(np.array(V_best)),
        "W": _to_list(np.array(W_best))
    }
