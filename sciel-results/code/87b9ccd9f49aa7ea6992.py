"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 138.0  # safety margin
    
    num_eqs = n * m * p  # 48
    
    def residual_matrix(U, V, W, r):
        """Return 3D array of residuals, shape (n,m,p)."""
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    Uc = U[i*m + j]
                    Vc = V[j*p + k]
                    Wc = W[k*n + i]
                    for c in range(r):
                        s += Uc[c] * Vc[c] * Wc[c]
                    res[i, j, k] = s - 1.0
        return res
    
    def check_exact(U, V, W, r, tol=1e-9):
        res = residual_matrix(U, V, W, r)
        return np.max(np.abs(res)) < tol, res
    
    def jacobian(U, V, W, r):
        """Jacobian: shape (num_eqs, 46*r)."""
        nU = n * m * r
        nV = m * p * r
        nW = p * n * r
        J = np.zeros((num_eqs, nU + nV + nW))
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        ui = i * m + j
                        vi = j * p + k
                        wi = k * n + i
                        # d(res)/d(U[ui,c]) = V[vi,c] * W[wi,c]
                        J[eq_idx, ui * r + c] = V[vi, c] * W[wi, c]
                        # d(res)/d(V[vi,c]) = U[ui,c] * W[wi,c]
                        J[eq_idx, nU + vi * r + c] = U[ui, c] * W[wi, c]
                        # d(res)/d(W[wi,c]) = U[ui,c] * V[vi,c]
                        J[eq_idx, nU + nV + wi * r + c] = U[ui, c] * V[vi, c]
                    eq_idx += 1
        return J
    
    def newton(U, V, W, r, max_iter=200):
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            res = residual_matrix(U, V, W, r).flatten()
            res_norm = np.max(np.abs(res))
            if res_norm < 1e-10:
                return True
            if res_norm < 1e-8:
                return True
            
            J = jacobian(U, V, W, r)
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                delta = np.linalg.lstsq(J, -res, rcond=None)[0]
            except:
                break
            
            # Line search
            best_t = 1.0
            best_norm = res_norm
            for t in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
                new_params = params + t * delta
                nU = n * m * r
                nV = m * p * r
                new_U = new_params[:nU].reshape(n*m, r)
                new_V = new_params[nU:nU+nV].reshape(m*p, r)
                new_W = new_params[nU+nV:].reshape(p*n, r)
                new_res = residual_matrix(new_U, new_V, new_W, r)
                new_norm = np.max(np.abs(new_res))
                if new_norm < best_norm:
                    best_norm = new_norm
                    best_t = t
            
            params = params + best_t * delta
            nU = n * m * r
            nV = m * p * r
            U = params[:nU].reshape(n*m, r)
            V = params[nU:nU+nV].reshape(m*p, r)
            W = params[nU+nV:].reshape(p*n, r)
        
        res = residual_matrix(U, V, W, r)
        return np.max(np.abs(res)) < 1e-8
    
    def search_rank(r, num_trials=40):
        if time.time() > deadline - 8.0:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 8.0:
                break
            
            # Various initialization strategies
            if trial % 4 == 0:
                # Small random
                U0 = np.random.uniform(-0.5, 0.5, (n*m, r))
                V0 = np.random.uniform(-0.5, 0.5, (m*p, r))
                W0 = np.random.uniform(-0.5, 0.5, (p*n, r))
            elif trial % 4 == 1:
                # Sparse random
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                for c in range(r):
                    # Random non-zero pattern
                    for idx in np.random.choice(n*m, size=max(1, n*m//2), replace=False):
                        U0[idx, c] = np.random.choice([-1.0, 1.0])
                    for idx in np.random.choice(m*p, size=max(1, m*p//2), replace=False):
                        V0[idx, c] = np.random.choice([-1.0, 1.0])
                    for idx in np.random.choice(p*n, size=max(1, p*n//2), replace=False):
                        W0[idx, c] = np.random.choice([-1.0, 1.0])
            elif trial % 4 == 2:
                # Block-diagonal inspired
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                # Try to cover equations with blocks
                for c in range(r):
                    i0 = c % n
                    j0 = (c // n) % m
                    U0[i0*m + j0, c] = 1.0
                    for k in range(p):
                        V0[j0*p + k, c] = np.random.choice([-0.5, 0.5, 1.0])
                        W0[k*n + i0, c] = np.random.choice([-0.5, 0.5, 1.0])
            else:
                # Mixed scale
                U0 = np.random.uniform(-1.0, 1.0, (n*m, r))
                V0 = np.random.uniform(-1.0, 1.0, (m*p, r))
                W0 = np.random.uniform(-1.0, 1.0, (p*n, r))
            
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            success = newton(U, V, W, r)
            
            if success:
                # Try to get clean half-integer solution
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                ok, _ = check_exact(U_r, V_r, W_r, r)
                if ok:
                    return U_r, V_r, W_r
                
                # Check original
                ok, _ = check_exact(U, V, W, r, tol=1e-10)
                if ok:
                    return U, V, W
                else:
                    # Return as-is if very close
                    ok2, _ = check_exact(U, V, W, r, tol=1e-8)
                    if ok2:
                        return U, V, W
        
        return None
    
    # Try increasing ranks
    for r in range(8, 49):
        if time.time() > deadline - 15.0:
            break
        result = search_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
