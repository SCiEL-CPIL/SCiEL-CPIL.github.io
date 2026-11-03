import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    deadline = time.time() + 130  # Safety margin
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def compute_residual(U, V, W):
        R = U.shape[1]
        S = np.zeros((nm, mp, pn))
        for r in range(R):
            S += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.sum((S - T)**2)

    def check_and_round(U, V, W):
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if exact
        R = U_r.shape[1]
        S = np.zeros((nm, mp, pn))
        for r in range(R):
            S += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
        
        if np.allclose(S, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def run_als(U, V, W, max_iters=30, tol=1e-10):
        prev_residual = np.inf
        for it in range(max_iters):
            R = U.shape[1]
            
            # Fix V, W, solve for U
            M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, R)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                U[a, :], _, _, _ = np.linalg.lstsq(M_VW, t_a, rcond=None)
            
            # Fix U, W, solve for V
            M_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, R)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], _, _, _ = np.linalg.lstsq(M_UW, t_b, rcond=None)
            
            # Fix U, V, solve for W
            M_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, R)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], _, _, _ = np.linalg.lstsq(M_UV, t_c, rcond=None)
            
            # Check convergence
            residual = compute_residual(U, V, W)
            if abs(prev_residual - residual) < tol:
                break
            prev_residual = residual
            
        return U, V, W, residual

    def polish(U, V, W, R, max_time=5.0):
        """Final high-precision polish using scipy minimize"""
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        total_size = size_U + size_V + size_W
        
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        params = np.concatenate([U_flat, V_flat, W_flat])
        
        def objective(params):
            U = params[:size_U].reshape(nm, R)
            V = params[size_U:size_U+size_V].reshape(mp, R)
            W = params[size_U+size_V:].reshape(pn, R)
            S = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((S - T)**2)
        
        start_time = time.time()
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 500})
        except Exception:
            return U, V, W
        
        if time.time() - start_time > max_time:
            return U, V, W
        
        if result.fun < 1e-12:
            U_pol = result.x[:size_U].reshape(nm, R)
            V_pol = result.x[size_U:size_U+size_V].reshape(mp, R)
            W_pol = result.x[size_U+size_V:].reshape(pn, R)
            return U_pol, V_pol, W_pol
        
        return U, V, W

    # Try ranks from 21 to 24
    for R in [21, 22, 23, 24]:
        if time.time() > deadline - 15:
            break
            
        # Number of restarts based on time remaining
        time_left = deadline - time.time()
        num_restarts = min(8, max(2, int(time_left / 15)))
        
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Different initialization strategies
            if restart == 0:
                # Random small
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart == 1:
                # Structured: start from schoolbook-like pattern
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                # Initialize with a subset of schoolbook columns
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < R:
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
                # Add random perturbation
                U += np.random.randn(nm, R) * 0.01
                V += np.random.randn(mp, R) * 0.01
                W += np.random.randn(pn, R) * 0.01
            elif restart == 2:
                # Random medium
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            else:
                # Random with varying scale
                scale = 0.1 * (1 + restart * 0.1)
                U = np.random.randn(nm, R) * scale
                V = np.random.randn(mp, R) * scale
                W = np.random.randn(pn, R) * scale
            
            # Run ALS
            U, V, W, residual = run_als(U, V, W, max_iters=40)
            
            # If residual is very small, try polish
            if residual < 1e-6:
                U, V, W = polish(U, V, W, R, max_time=5.0)
                residual = compute_residual(U, V, W)
            
            # Check and round
            result = check_and_round(U, V, W)
            if result is not None:
                U_r, V_r, W_r = result
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also check without rounding
            if residual < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If R==24, return schoolbook immediately
        if R == 24:
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
