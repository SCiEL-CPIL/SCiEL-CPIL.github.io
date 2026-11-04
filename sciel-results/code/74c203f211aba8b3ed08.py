import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = 3, 4, 5
    
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Schoolbook: rank = n*m*p = 60
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_sb
    
    T = np.ones((n, m, p))
    
    def verify(U, V, W, tol=1e-6):
        """Verify that U, V, W decompose the tensor T."""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(S, T, atol=tol)
    
    def residuals(params, r):
        """Residuals for least_squares: T - S where S[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]"""
        U_flat = params[:n * m * r]
        V_flat = params[n * m * r : n * m * r + m * p * r]
        W_flat = params[n * m * r + m * p * r : n * m * r + m * p * r + p * n * r]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        res = (S - T).flatten()
        return res
    
    # Strategy: Try to find a low-rank decomposition using structured approaches
    # and numerical optimization with multiple restarts.
    
    # Lower bound for rank: max(n*m, m*p, p*n) is not quite right.
    # The tensor has dimensions n*m*p = 60 entries.
    # Known lower bounds: rank >= max(n, m, p) * min(n, m, p) ... not standard.
    # A simple lower bound: rank >= n*m*p / max(n*m, m*p, p*n) = 60/20 = 3, but that's weak.
    # Actually, the standard lower bound is: rank >= max(n*m, m*p, p*n) / min(n, m, p)
    # For 3x4x5: max(12, 20, 15)/3 = 20/3 ≈ 6.67, so rank >= 7.
    # But we know rank >= n+m+p-2 for some cases... not standard here.
    # Let's just try from a reasonable lower bound upward.
    
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    # Known: 2x2x2 has rank 7. For larger tensors, we can try various approaches.
    # Start from 20 (a reasonable lower bound) and go up to 60.
    
    found = False
    
    for target_rank in range(20, 61):
        if time.time() > deadline:
            break
        
        # Multiple restart strategies for this rank
        max_restarts = 30 if target_rank < 30 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 5 == 0:
                # Random Gaussian
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart % 5 == 1:
                # Random uniform
                U_init = np.random.rand(n * m, target_rank) * 2 - 1
                V_init = np.random.rand(m * p, target_rank) * 2 - 1
                W_init = np.random.rand(p * n, target_rank) * 2 - 1
            elif restart % 5 == 2:
                # Binary initialization
                U_init = np.random.choice([-1, 1], size=(n * m, target_rank)).astype(float)
                V_init = np.random.choice([-1, 1], size=(m * p, target_rank)).astype(float)
                W_init = np.random.choice([-1, 1], size=(p * n, target_rank)).astype(float)
            elif restart % 5 == 3:
                # Sparse initialization
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    idx_u = np.random.choice(n * m, size=max(1, n * m // 3), replace=False)
                    idx_v = np.random.choice(m * p, size=max(1, m * p // 3), replace=False)
                    idx_w = np.random.choice(p * n, size=max(1, p * n // 3), replace=False)
                    U_init[idx_u, r] = np.random.choice([-1, 1])
                    V_init[idx_v, r] = np.random.choice([-1, 1])
                    W_init[idx_w, r] = np.random.choice([-1, 1])
            else:
                # Structured: use Kronecker-like patterns
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Each term activates a random combination
                    i_r = r % n
                    j_r = (r // n) % m
                    k_r = (r // (n * m)) % p
                    U_init[i_r * m + j_r, r] = 1.0
                    V_init[j_r * p + k_r, r] = 1.0
                    W_init[k_r * n + i_r, r] = 1.0
                    # Add some noise
                    if r < target_rank // 2:
                        U_init[np.random.choice(n * m, size=2, replace=False), r] += np.random.randn(2) * 0.1
                        V_init[np.random.choice(m * p, size=2, replace=False), r] += np.random.randn(2) * 0.1
                        W_init[np.random.choice(p * n, size=2, replace=False), r] += np.random.randn(2) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                result = least_squares(
                    residuals, x0, args=(target_rank,),
                    method='trf', max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                
                if result.cost < 1e-8:
                    U_opt = result.x[:n * m * target_rank].reshape(n, m, target_rank)
                    V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m, p, target_rank)
                    W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p, n, target_rank)
                    
                    # Try to snap to half-integers or integers
                    # First try half-integers
                    U_r = np.round(U_opt * 2) / 2.0
                    V_r = np.round(V_opt * 2) / 2.0
                    W_r = np.round(W_opt * 2) / 2.0
                    
                    if verify(U_r, V_r, W_r, tol=1e-4):
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        best_rank = target_rank
                        found = True
                        break
                    
                    # Try integers
                    U_i = np.round(U_opt)
                    V_i = np.round(V_opt)
                    W_i = np.round(W_opt)
                    if verify(U_i, V_i, W_r, tol=1e-4):
                        best_U = U_i
                        best_V = V_i
                        best_W = W_i
                        best_rank = target_rank
                        found = True
                        break
                    
                    # Try quarter-integers
                    U_q = np.round(U_opt * 4) / 4.0
                    V_q = np.round(V_opt * 4) / 4.0
                    W_q = np.round(W_opt * 4) / 4.0
                    if verify(U_q, V_q, W_q, tol=1e-4):
                        best_U = U_q
                        best_V = V_q
                        best_W = W_q
                        best_rank = target_rank
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    # If we found a better rank, use it; otherwise fall back to schoolbook
    if not verify(best_U, best_V, best_W, tol=1e-4):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
