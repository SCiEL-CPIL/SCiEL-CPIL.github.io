import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    total_entries = n * m * p  # 18

    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute the 18 tensor entries
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                eqs.append((u, v, w))

    # Precompute index lists for ALS
    u_indices = {u: [] for u in range(U_size)}
    v_indices = {v: [] for v in range(V_size)}
    w_indices = {w: [] for w in range(W_size)}
    for idx, (u, v, w) in enumerate(eqs):
        u_indices[u].append((v, w, idx))
        v_indices[v].append((u, w, idx))
        w_indices[w].append((u, v, idx))

    # Schoolbook decomposition (rank 18)
    rank_school = total_entries
    U_school = np.zeros((U_size, rank_school))
    V_school = np.zeros((V_size, rank_school))
    W_school = np.zeros((W_size, rank_school))
    for col, (u, v, w) in enumerate(eqs):
        U_school[u, col] = 1
        V_school[v, col] = 1
        W_school[w, col] = 1

    def verify_integer(U_int, V_int, W_int):
        """Check exact decomposition using integer arithmetic (entries are 2*half-int)."""
        for (u, v, w) in eqs:
            s = 0
            for r in range(U_int.shape[1]):
                s += U_int[u, r] * V_int[v, r] * W_int[w, r]
            if s != 8:  # 2^3 * 1 = 8
                return False
        return True

    def calculate_error_fast(U, V, W):
        """Fast error computation using precomputed structure."""
        err = 0.0
        for (u, v, w) in eqs:
            s = np.dot(U[u, :], V[v, :] * W[w, :])
            err += (s - 1.0) ** 2
        return err

    def snap_and_check(U, V, W):
        """Snap to half-integers and verify."""
        U_int = np.round(U * 2).astype(np.int64)
        V_int = np.round(V * 2).astype(np.int64)
        W_int = np.round(W * 2).astype(np.int64)
        if verify_integer(U_int, V_int, W_int):
            return U_int / 2.0, V_int / 2.0, W_int / 2.0
        return None

    def als_decompose(U_init, V_init, W_init, R, max_iters=150, tol=1e-12):
        """ALS with symmetry breaking: fix U[0,0]=1."""
        U = U_init.copy().astype(np.float64)
        V = V_init.copy().astype(np.float64)
        W = W_init.copy().astype(np.float64)
        
        # Enforce symmetry breaking
        U[0, 0] = 1.0
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U rows (skip row 0, col 0 which is fixed)
            for u in range(U_size):
                if not u_indices[u]:
                    continue
                if u == 0:
                    # For row 0, only optimize cols 1..R-1, keep col 0 = 1
                    B = np.zeros((len(u_indices[u]), R - 1))
                    for k, (v, w, idx) in enumerate(u_indices[u]):
                        B[k, :] = (V[v, 1:] * W[w, 1:])
                    target = np.ones(len(u_indices[u])) - V[v_indices[0][0][0], 0] * W[w_indices[0][0][0], 0]
                    if len(B) > 0 and B.shape[1] > 0:
                        U_u, _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                        U[0, 1:] = U_u
                    U[0, 0] = 1.0
                else:
                    B = np.zeros((len(u_indices[u]), R))
                    for k, (v, w, idx) in enumerate(u_indices[u]):
                        B[k, :] = V[v, :] * W[w, :]
                    target = np.ones(len(u_indices[u]))
                    if len(B) > 0:
                        U_u, _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                        U[u, :] = U_u
            
            # Update V rows
            for v in range(V_size):
                if not v_indices[v]:
                    continue
                C = np.zeros((len(v_indices[v]), R))
                for k, (u, w, idx) in enumerate(v_indices[v]):
                    C[k, :] = U[u, :] * W[w, :]
                target = np.ones(len(v_indices[v]))
                if len(C) > 0:
                    V_v, _, _, _ = np.linalg.lstsq(C, target, rcond=None)
                    V[v, :] = V_v
            
            # Update W rows
            for w in range(W_size):
                if not w_indices[w]:
                    continue
                D = np.zeros((len(w_indices[w]), R))
                for k, (u, v, idx) in enumerate(w_indices[w]):
                    D[k, :] = U[u, :] * V[v, :]
                target = np.ones(len(w_indices[w]))
                if len(D) > 0:
                    W_w, _, _, _ = np.linalg.lstsq(D, target, rcond=None)
                    W[w, :] = W_w
            
            # Recalculate U[0,0] consistency (it's fixed, but check)
            U[0, 0] = 1.0
            
            err = calculate_error_fast(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_err - err) < tol and it > 10:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        U[0, 0] = 1.0
        
        # Try to snap
        sol = snap_and_check(U, V, W)
        if sol is not None:
            return sol
        
        return None

    def optimize_with_scipy(U_init, V_init, W_init, R, time_budget=5.0):
        """Use scipy optimizers with symmetry breaking."""
        if time.time() > deadline:
            return None
        
        # Parameter vector: U[1:, :] (6R-1), V (9R), W (6R)
        # Total: 21R - 1 parameters
        n_params = 21 * R - 1
        
        def unpack(params):
            U_flat = np.empty(U_size * R)
            U_flat[0] = 1.0
            U_flat[1:] = params[:U_size * R - 1]
            U = U_flat.reshape(U_size, R)
            V = params[U_size * R - 1:U_size * R - 1 + V_size * R].reshape(V_size, R)
            W = params[U_size * R - 1 + V_size * R:].reshape(W_size, R)
            return U, V, W
        
        def objective(params):
            U, V, W = unpack(params)
            return calculate_error_fast(U, V, W)
        
        # Initial parameter vector
        U_flat = U_init.flatten()
        U_flat[0] = 1.0
        params = np.concatenate([U_flat[1:], V_init.flatten(), W_init.flatten()])
        
        # Stage 1: Nelder-Mead
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
        
        if time.time() > deadline:
            return None
        
        if res.fun < 1e-4:
            # Stage 2: Powell
            res2 = minimize(objective, res.x, method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < res.fun:
                res = res2
        
        if res.fun < 1e-5:
            U, V, W = unpack(res.x)
            sol = snap_and_check(U, V, W)
            if sol is not None:
                return sol
            
            # Try ALS refinement from this point
            sol = als_decompose(U, V, W, R, max_iters=50)
            if sol is not None:
                return sol
        
        return None

    def try_rank(R, num_restarts=30, time_limit=15.0):
        """Try to find a rank-R decomposition."""
        start = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start > time_limit or time.time() > deadline:
                break
            
            np.random.seed(restart * 1000 + R * 7)
            strategy = restart % 4
            
            if strategy == 0:
                # Random small initialization
                U_init = np.random.randn(U_size, R) * 0.3
                V_init = np.random.randn(V_size, R) * 0.3
                W_init = np.random.randn(W_size, R) * 0.3
            elif strategy == 1:
                # Sparse half-integer initialization
                U_init = np.zeros((U_size, R))
                V_init = np.zeros((V_size, R))
                W_init = np.zeros((W_size, R))
                for r in range(R):
                    U_init[np.random.randint(U_size), r] = np.random.choice([-1, -0.5, 0.5, 1])
                    V_init[np.random.randint(V_size), r] = np.random.choice([-1, -0.5, 0.5, 1])
                    W_init[np.random.randint(W_size), r] = np.random.choice([-1, -0.5, 0.5, 1])
            elif strategy == 2:
                # Structured from tensor entries
                U_init = np.zeros((U_size, R))
                V_init = np.zeros((V_size, R))
                W_init = np.zeros((W_size, R))
                for r in range(min(R, len(eqs))):
                    u, v, w = eqs[r]
                    U_init[u, r] = 1
                    V_init[v, r] = 1
                    W_init[w, r] = 1
                if R > len(eqs):
                    for r in range(len(eqs), R):
                        U_init[np.random.randint(U_size), r] = np.random.choice([-0.5, 0.5])
                        V_init[np.random.randint(V_size), r] = np.random.choice([-0.5, 0.5])
                        W_init[np.random.randint(W_size), r] = np.random.choice([-0.5, 0.5])
            else:
                # Mixed: some structured, some random
                U_init = np.zeros((U_size, R))
                V_init = np.zeros((V_size, R))
                W_init = np.zeros((W_size, R))
                n_struct = min(R // 2, len(eqs))
                for r in range(n_struct):
                    u, v, w = eqs[r]
                    U_init[u, r] = 1
                    V_init[v, r] = 1
                    W_init[w, r] = 1
                for r in range(n_struct, R):
                    U_init[np.random.randint(U_size), r] = np.random.choice([-1, -0.5, 0.5, 1])
                    V_init[np.random.randint(V_size), r] = np.random.choice([-1, -0.5, 0.5, 1])
                    W_init[np.random.randint(W_size), r] = np.random.choice([-1, -0.5, 0.5, 1])
            
            # Choose optimizer based on rank
            if R <= 13:
                # Lower ranks: try scipy first (more careful), then ALS
                res = optimize_with_scipy(U_init, V_init, W_init, R, time_budget=4.0)
                if res is not None:
                    return res
                res = als_decompose(U_init, V_init, W_init, R, max_iters=100)
                if res is not None:
                    return res
            else:
                # Higher ranks: ALS is faster
                res = als_decompose(U_init, V_init, W_init, R, max_iters=100)
                if res is not None:
                    return res
                res = optimize_with_scipy(U_init, V_init, W_init, R, time_budget=3.0)
                if res is not None:
                    return res
        
        return None

    # Try ranks from low to high, spending more time on lower ranks
    rank_time_allocation = {
        11: (40, 50),   # time_limit, num_restarts
        12: (30, 40),
        13: (20, 30),
        14: (15, 25),
        15: (10, 20),
        16: (8, 15),
        17: (5, 10),
    }
    
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        if time.time() > deadline - 5:
            break
        
        time_limit, num_restarts = rank_time_allocation[target_rank]
        res = try_rank(target_rank, num_restarts=num_restarts, time_limit=time_limit)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
