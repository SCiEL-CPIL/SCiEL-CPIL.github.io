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
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((S - T)**2)

    def check_exact(U, V, W, atol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        R = U.shape[1]
        S = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(S, T, atol=atol):
            return True
        return False

    def round_to_half(U, V, W):
        """Round to nearest half-integer and check exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if check_exact(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def run_als(U, V, W, max_iters=50, tol=1e-12):
        """Alternating least squares with proper lstsq usage."""
        prev_residual = np.inf
        for it in range(max_iters):
            R = U.shape[1]
            
            # Fix V, W, solve for U
            # For each a, solve: sum_r U[a,r] * (V[:,r] outer W[:,r]) = T[a,:,:]
            # This is: M_VW * U[a,:] = T[a,:,:].reshape(-1)
            M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, R)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M_VW, t_a, rcond=None)
                U[a, :] = sol
            
            # Fix U, W, solve for V
            M_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, R)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M_UW, t_b, rcond=None)
                V[b, :] = sol
            
            # Fix U, V, solve for W
            M_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, R)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M_UV, t_c, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            residual = compute_residual(U, V, W)
            if abs(prev_residual - residual) < tol or residual < 1e-14:
                break
            prev_residual = residual
            
        return U, V, W, residual

    def polish(U, V, W, R, max_time=8.0):
        """Final high-precision polish using scipy minimize with Nelder-Mead fallback."""
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
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
        
        # Try L-BFGS-B first
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 1000})
            if result.fun < 1e-14:
                U_pol = result.x[:size_U].reshape(nm, R)
                V_pol = result.x[size_U:size_U+size_V].reshape(mp, R)
                W_pol = result.x[size_U+size_V:].reshape(pn, R)
                return U_pol, V_pol, W_pol
        except Exception:
            pass
        
        if time.time() - start_time > max_time * 0.7:
            return U, V, W
        
        # Try Nelder-Mead as fallback
        try:
            result2 = minimize(objective, params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14})
            if result2.fun < 1e-14:
                U_pol = result2.x[:size_U].reshape(nm, R)
                V_pol = result2.x[size_U:size_U+size_V].reshape(mp, R)
                W_pol = result2.x[size_U+size_V:].reshape(pn, R)
                return U_pol, V_pol, W_pol
        except Exception:
            pass
        
        return U, V, W

    def schoolbook():
        """Return the schoolbook rank-24 decomposition."""
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

    # Try ranks from 20 to 24
    for R in [20, 21, 22, 23, 24]:
        if time.time() > deadline - 20:
            break
            
        # Number of restarts based on time remaining
        time_left = deadline - time.time()
        num_restarts = min(12, max(3, int(time_left / 10)))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            # Different initialization strategies
            if restart == 0:
                # Random small
                np.random.seed(42 + R * 100 + restart)
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart == 1:
                # Structured: start from schoolbook-like pattern with perturbation
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
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
                np.random.seed(43 + R * 100 + restart)
                U += np.random.randn(nm, R) * 0.05
                V += np.random.randn(mp, R) * 0.05
                W += np.random.randn(pn, R) * 0.05
            elif restart == 2:
                # Random medium
                np.random.seed(44 + R * 100 + restart)
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            elif restart == 3:
                # Sparse random initialization
                np.random.seed(45 + R * 100 + restart)
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
                # Sparsify
                mask_U = np.random.rand(nm, R) < 0.3
                mask_V = np.random.rand(mp, R) < 0.3
                mask_W = np.random.rand(pn, R) < 0.3
                U *= mask_U
                V *= mask_V
                W *= mask_W
            else:
                # Random with varying scale
                np.random.seed(46 + R * 100 + restart)
                scale = 0.1 + 0.2 * restart
                U = np.random.randn(nm, R) * scale
                V = np.random.randn(mp, R) * scale
                W = np.random.randn(pn, R) * scale
            
            # Run ALS
            U, V, W, residual = run_als(U, V, W, max_iters=60)
            
            # If residual is small, try polish
            if residual < 1e-8:
                U, V, W = polish(U, V, W, R, max_time=6.0)
                residual = compute_residual(U, V, W)
            
            # Check exactness
            if residual < 1e-12:
                # Try rounding to half-integers
                rounded = round_to_half(U, V, W)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # Return as-is if exact
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If R==24, return schoolbook immediately
        if R == 24:
            return schoolbook()

    # Fallback: schoolbook
    return schoolbook()
