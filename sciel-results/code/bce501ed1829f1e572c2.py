import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    # The rank-17 solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    # Precompute indices for residual calculation
    # Non-zero indices
    nz_u, nz_v, nz_w = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_u.append(i * m + j)
                nz_v.append(j * p + k)
                nz_w.append(k * n + i)
    nz_u = np.array(nz_u)
    nz_v = np.array(nz_v)
    nz_w = np.array(nz_w)
    
    # Zero indices
    all_u = np.arange(n * m)
    all_v = np.arange(m * p)
    all_w = np.arange(p * n)
    # Create a set of non-zero tuples for fast lookup
    nz_set = set(zip(nz_u, nz_v, nz_w))
    zero_u, zero_v, zero_w = [], [], []
    for u in all_u:
        for v in all_v:
            for w in all_w:
                if (u, v, w) not in nz_set:
                    zero_u.append(u)
                    zero_v.append(v)
                    zero_w.append(w)
    zero_u = np.array(zero_u)
    zero_v = np.array(zero_v)
    zero_w = np.array(zero_w)
    
    def residual_full(U, V, W):
        """Compute the residual including zero constraints."""
        # Non-zero part
        # val_nz = np.einsum('ir, jr, kr -> ijk', U[nz_u], V[nz_v], W[nz_w])
        # This is tricky with repeated indices if not careful.
        # U[nz_u] is shape (18, R). V[nz_v] is (18, R). W[nz_w] is (18, R).
        # We want sum_r U[nz_u[i], r] * V[nz_v[i], r] * W[nz_w[i], r] for each i.
        val_nz = np.sum(U[nz_u] * V[nz_v] * W[nz_w], axis=1)
        res_nz = np.sum((val_nz - 1.0) ** 2)
        
        # Zero part
        # To save time, we might sample zeros or use a smaller weight
        # But for exactness, we need them to be close to 0.
        val_zero = np.sum(U[zero_u] * V[zero_v] * W[zero_w], axis=1)
        res_zero = np.sum(val_zero ** 2)
        
        # Weight the zero part to balance with non-zero part
        # 18 non-zeros, ~306 zeros.
        # If we weight zeros by 1/10, the total scale is similar.
        return res_nz + 0.1 * res_zero

    def residual_grad_full(U, V, W):
        """Compute the gradient of the residual including zero constraints."""
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        # Non-zero part
        val_nz = np.sum(U[nz_u] * V[nz_v] * W[nz_w], axis=1)
        diff_nz = 2 * (val_nz - 1.0)
        
        # d/dU = 2 * diff * V * W
        # We need to accumulate into U_grad
        # U_grad[nz_u[i]] += diff_nz[i] * V[nz_v[i]] * W[nz_w[i]]
        # Vectorized update is hard with repeated indices.
        # Loop is safer for gradient accumulation.
        for i in range(len(nz_u)):
            u_idx = nz_u[i]
            v_idx = nz_v[i]
            w_idx = nz_w[i]
            diff = diff_nz[i]
            vw = V[v_idx] * W[w_idx]
            U_grad[u_idx] += diff * vw
            V_grad[v_idx] += diff * U[u_idx] * W[w_idx]
            W_grad[w_idx] += diff * U[u_idx] * V[v_idx]
            
        # Zero part (weighted by 0.1)
        # val_zero = sum U[zero_u[i]] * V[zero_v[i]] * W[zero_w[i]]
        # diff_zero = 2 * 0.1 * val_zero
        # Loop for zeros is expensive (306 iterations).
        # We can subsample zeros if time is tight, but let's try full first.
        val_zero = np.sum(U[zero_u] * V[zero_v] * W[zero_w], axis=1)
        diff_zero = 2 * 0.1 * val_zero
        
        for i in range(len(zero_u)):
            u_idx = zero_u[i]
            v_idx = zero_v[i]
            w_idx = zero_w[i]
            diff = diff_zero[i]
            vw = V[v_idx] * W[w_idx]
            U_grad[u_idx] += diff * vw
            V_grad[v_idx] += diff * U[u_idx] * W[w_idx]
            W_grad[w_idx] += diff * U[u_idx] * V[v_idx]
            
        return U_grad, V_grad, W_grad

    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W = x[(n*m+m*p)*rank:].reshape(p*n, rank)
        return U, V, W

    best_solution = (U17, V17, W17)
    # Calculate residual of the known solution using the full metric
    best_res = residual_full(U17, V17, W17)
    
    # Try to find rank-16 solution using L-BFGS-B
    R = 16
    
    def try_lbfgs(init_U, init_V, init_W, max_iter=500, rank=R):
        nonlocal best_res, best_solution
        x0 = flatten(init_U, init_V, init_W)
        
        def fun(x):
            U, V, W = unflatten(x, rank)
            return residual_full(U, V, W)
        
        def jac(x):
            U, V, W = unflatten(x, rank)
            Ug, Vg, Wg = residual_grad_full(U, V, W)
            return flatten(Ug, Vg, Wg)
        
        try:
            result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                             options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-12})
            if result.fun < best_res:
                U, V, W = unflatten(result.x, rank)
                best_res = result.fun
                best_solution = (U.copy(), V.copy(), W.copy())
            return result.fun
        except Exception:
            return np.inf

    # Try each column removal with L-BFGS-B
    for remove_col in range(17):
        if time.time() > deadline:
            break
        
        cols = [r for r in range(17) if r != remove_col]
        U = U17[:, cols]
        V = V17[:, cols]
        W = W17[:, cols]
        
        try_lbfgs(U, V, W, max_iter=500, rank=16)
        
        if best_res < 1e-8:
            break
    
    # Try random initializations with L-BFGS-B
    np.random.seed(42)
    num_random_attempts = 50
    for attempt in range(num_random_attempts):
        if time.time() > deadline:
            break
        
        scale = 0.1 + 0.2 * (attempt % 5)
        U = np.random.randn(n * m, R) * scale
        V = np.random.randn(m * p, R) * scale
        W = np.random.randn(p * n, R) * scale
        
        try_lbfgs(U, V, W, max_iter=300, rank=16)
        
        if best_res < 1e-8:
            break
    
    # If we found a good rank-16 solution, try to round to half-integers
    if best_res < 1e-6:
        U_best, V_best, W_best = best_solution
        best_round_res = np.inf
        best_rounded = None
        
        for factor in [2, 4]:
            U_round = np.round(U_best * factor) / factor
            V_round = np.round(V_best * factor) / factor
            W_round = np.round(W_best * factor) / factor
            
            # Check full residual for rounded solution
            res_round = residual_full(U_round, V_round, W_round)
            if res_round < best_round_res:
                best_round_res = res_round
                best_rounded = (U_round, V_round, W_round)
        
        if best_rounded is not None and best_round_res < 1e-6:
            U_round, V_round, W_round = best_rounded
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
