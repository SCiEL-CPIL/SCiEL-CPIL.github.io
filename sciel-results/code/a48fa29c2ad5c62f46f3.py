import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Uses Alternating Least Squares (ALS) with random restarts to search for lower-rank decompositions.
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with naive rank-24 decomposition
    rank_naive = nm * p  # 24
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_naive[a, r] = 1.0
                V_naive[b, r] = 1.0
                W_naive[c, r] = 1.0
    
    best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    best_rank = rank_naive
    
    # Deadline guard
    deadline = time.time() + 140  # Leave 10s buffer for safety

    def check_and_snap(U, V, W, T, tol=1e-8):
        """Check if decomposition is exact and try to snap to half-integers."""
        T_approx = np.zeros_like(T)
        for r in range(U.shape[1]):
            T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        residual = np.max(np.abs(T - T_approx))
        if residual < tol:
            # Try snapping to half-integers
            U_snap = np.round(U * 2) / 2.0
            V_snap = np.round(V * 2) / 2.0
            W_snap = np.round(W * 2) / 2.0
            
            T_approx_snap = np.zeros_like(T)
            for r in range(U_snap.shape[1]):
                T_approx_snap += np.einsum('a,b,c->abc', U_snap[:,r], V_snap[:,r], W_snap[:,r])
            
            if np.max(np.abs(T - T_approx_snap)) < tol:
                return True, U_snap, V_snap, W_snap
            
            # If snapping fails, return the float solution
            return True, U, V, W
        return False, None, None, None

    def als_decompose(T, nm, mp, pn, rank, n_iter=100, tol=1e-12):
        """Run ALS for a given rank."""
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        prev_err = float('inf')
        for it in range(n_iter):
            # Update U
            for a in range(nm):
                # T[a, :, :] = sum_r U[a, r] * V[:, r] * W[:, r]
                # Let M_r = V[:, r] * W[:, r] (size mp x pn)
                # T[a, b, c] = sum_r U[a, r] M_r[b, c]
                # This is a linear system for U[a, :]
                # A is (mp*pn) x rank
                # t is T[a, :, :] flattened
                A = np.zeros((mp * pn, rank))
                for r in range(rank):
                    M_r = np.outer(V[:, r], W[:, r])
                    A[:, r] = M_r.flatten()
                t = T[a, :, :].flatten()
                # Solve least squares
                U[a, :], residuals, rank_sol, sv = np.linalg.lstsq(A, t, rcond=None)
            
            # Update V
            for b in range(mp):
                A = np.zeros((nm * pn, rank))
                for r in range(rank):
                    M_r = np.outer(U[:, r], W[:, r])
                    A[:, r] = M_r.flatten()
                t = T[:, b, :].flatten()
                V[b, :], residuals, rank_sol, sv = np.linalg.lstsq(A, t, rcond=None)
                
            # Update W
            for c in range(pn):
                A = np.zeros((nm * mp, rank))
                for r in range(rank):
                    M_r = np.outer(U[:, r], V[:, r])
                    A[:, r] = M_r.flatten()
                t = T[:, :, c].flatten()
                W[c, :], residuals, rank_sol, sv = np.linalg.lstsq(A, t, rcond=None)
            
            # Check convergence
            T_approx = np.zeros_like(T)
            for r in range(rank):
                T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.sum((T - T_approx)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Search for lower rank decompositions
    # Theoretical lower bound is often cited as 14 for 2x3x4. We start from 14.
    for target_rank in range(14, rank_naive):
        if time.time() > deadline:
            break
        
        # Increase restarts for lower ranks as they are harder
        n_restarts = 5 if target_rank < 16 else 3
        
        found_valid = False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + restart)
            
            U, V, W = als_decompose(T, nm, mp, pn, target_rank, n_iter=200)
            
            valid, U_final, V_final, W_final = check_and_snap(U, V, W, T)
            
            if valid:
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                best_rank = target_rank
                found_valid = True
                break
        
        if found_valid:
            continue
        else:
            # If we fail to find a solution at this rank, we might have hit a local minimum barrier.
            # We continue to the next rank.
            pass
    
    return best_result
