import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    # Schoolbook rank-24 solution (always valid fallback)
    rank_seed = n * m * p
    U_seed = np.zeros((nm, rank_seed), dtype=np.float64)
    V_seed = np.zeros((mp, rank_seed), dtype=np.float64)
    W_seed = np.zeros((pn, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Target tensor for full verification
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Constraint list: for each (i,j,k) we need sum_k U[i*m+j,k]*V[j*p+k,k]*W[k*n+i,k] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i * m + j, j * p + k, k * n + i))
    num_constraints = len(constraints)
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def full_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target) ** 2)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def round_and_verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if full_error(U_r, V_r, W_r) < tol:
            return U_r, V_r, W_r
        return None

    def verify_exact(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (all entries of tensor match)."""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(T_approx - target)) < tol

    # --- Smooth objective + analytic gradient over the 24 scalar constraints ---
    def objective_and_grad(params, r):
        U = params[:r * nm].reshape(nm, r)
        V = params[r * nm:r * (nm + mp)].reshape(mp, r)
        W = params[r * (nm + mp):].reshape(pn, r)
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        diff = vals - 1.0
        obj = np.sum(diff ** 2)
        
        dU_flat = np.zeros(r * nm)
        dV_flat = np.zeros(r * mp)
        dW_flat = np.zeros(r * pn)
        
        Vv = V[v_idxs, :]
        Ww = W[w_idxs, :]
        Uu = U[u_idxs, :]
        
        contrib_U = 2.0 * diff[:, None] * Vv * Ww
        idx_U = (u_idxs[:, None] * r + np.arange(r)[None, :]).ravel()
        np.add.at(dU_flat, idx_U, contrib_U.ravel())
        
        contrib_V = 2.0 * diff[:, None] * Uu * Ww
        idx_V = (v_idxs[:, None] * r + np.arange(r)[None, :]).ravel()
        np.add.at(dV_flat, idx_V, contrib_V.ravel())
        
        contrib_W = 2.0 * diff[:, None] * Uu * Vv
        idx_W = (w_idxs[:, None] * r + np.arange(r)[None, :]).ravel()
        np.add.at(dW_flat, idx_W, contrib_W.ravel())
        
        grad = np.concatenate([dU_flat, dV_flat, dW_flat])
        return obj, grad

    def try_rank_smooth(r, num_restarts=30, max_iters=150):
        if time.time() > deadline - 10.0:
            return None
        size = r * (nm + mp + pn)
        best_err = float('inf')
        best_params = None

        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                return None
            np.random.seed(12345 + restart * 7 + r)

            U0 = np.zeros((nm, r))
            V0 = np.zeros((mp, r))
            W0 = np.zeros((pn, r))
            
            # Structured initialization strategy
            if restart % 3 == 0:
                # Assign each column to a random constraint (original strategy)
                assigned = np.random.randint(0, num_constraints, size=r)
                for c in range(r):
                    cc = assigned[c]
                    u_c, v_c, w_c = constraints[cc]
                    U0[u_c, c] = 1.0
                    V0[v_c, c] = 1.0
                    W0[w_c, c] = 1.0
                scale = 0.1 + 0.05 * (restart % 4)
                U0 += np.random.randn(nm, r) * scale
                V0 += np.random.randn(mp, r) * scale
                W0 += np.random.randn(pn, r) * scale
            elif restart % 3 == 1:
                # Sparse random (inspiration 2)
                U0 = np.random.randn(nm, r) * 0.7
                V0 = np.random.randn(mp, r) * 0.7
                W0 = np.random.randn(pn, r) * 0.7
                mask_U = np.random.rand(nm, r) > 0.6
                mask_V = np.random.rand(mp, r) > 0.6
                mask_W = np.random.rand(pn, r) > 0.6
                U0[mask_U] = 0
                V0[mask_V] = 0
                W0[mask_W] = 0
            else:
                # Random with bias toward small values
                U0 = np.random.randn(nm, r) * 0.5
                V0 = np.random.randn(mp, r) * 0.5
                W0 = np.random.randn(pn, r) * 0.5
                U0[np.abs(U0) < 0.2] = 0
                V0[np.abs(V0) < 0.2] = 0
                W0[np.abs(W0) < 0.2] = 0

            params = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])

            def objective(params):
                return objective_and_grad(params, r)[0]

            def gradient(params):
                return objective_and_grad(params, r)[1]

            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': max_iters, 'ftol': 1e-12,
                                        'gtol': 1e-10, 'maxfun': 500})
            except Exception:
                continue

            if time.time() > deadline - 10.0:
                return None

            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                if err < 1e-8:
                    U = res.x[:r * nm].reshape(nm, r)
                    V = res.x[r * nm:r * (nm + mp)].reshape(mp, r)
                    W = res.x[r * (nm + mp):].reshape(pn, r)
                    
                    # Try rounding
                    result = round_and_verify(U, V, W)
                    if result:
                        return result
                    
                    # Try exact verification without rounding (in case entries are already half-integers)
                    if verify_exact(U, V, W):
                        return U, V, W

        return None

    # --- High-precision Newton polish ---
    def newton_polish(U, V, W, r, max_iters=30):
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            size = r * (nm + mp + pn)
            J = np.zeros((num_constraints, size))
            vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
            diff = vals - 1.0
            
            for c in range(num_constraints):
                u_c, v_c, w_c = constraints[c]
                for k in range(r):
                    J[c, u_c * r + k] = V[v_c, k] * W[w_c, k]
                    J[c, r * nm + v_c * r + k] = U[u_c, k] * W[w_c, k]
                    J[c, r * (nm + mp) + w_c * r + k] = U[u_c, k] * V[v_c, k]
            
            try:
                delta, _, _, _ = np.linalg.lstsq(J, -diff, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            step = 1.0
            improved = False
            for _ in range(20):
                U_new = (U.ravel() + step * delta[:r * nm]).reshape(nm, r)
                V_new = (V.ravel() + step * delta[r * nm:r * (nm + mp)]).reshape(mp, r)
                W_new = (W.ravel() + step * delta[r * (nm + mp):]).reshape(pn, r)
                new_err = error_function(U_new, V_new, W_new)
                if new_err < error_function(U, V, W):
                    U, V, W = U_new, V_new, W_new
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                break
            if error_function(U, V, W) < 1e-14:
                break
        return U, V, W

    # Try ranks from 19 up to 23
    for r in range(19, 24):
        if time.time() > deadline - 20.0:
            break
        sol = try_rank_smooth(r, num_restarts=40, max_iters=150)
        if sol:
            U_sol, V_sol, W_sol = sol
            # Apply Newton polish to get closer to exact half-integers
            U_pol, V_pol, W_pol = newton_polish(U_sol, V_sol, W_sol, r)
            
            # Final rounding and verification
            U_fin = np.round(U_pol * 2) / 2
            V_fin = np.round(V_pol * 2) / 2
            W_fin = np.round(W_pol * 2) / 2
            
            if verify_exact(U_fin, V_fin, W_fin):
                best_U, best_V, best_W = U_fin, V_fin, W_fin
                best_rank = r
                break
            elif verify_exact(U_pol, V_pol, W_pol):
                best_U, best_V, best_W = U_pol, V_pol, W_pol
                best_rank = r
                break

    # Final safety: verify the returned solution
    if not verify_exact(best_U, best_V, best_W):
        best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
        best_rank = rank_seed

    # Ensure entries are integer or half-integer
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
