import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    rank22 = 22
    U22 = np.zeros((n * m, rank22))
    V22 = np.zeros((m * p, rank22))
    W22 = np.zeros((p * n, rank22))

    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U22[idx, col] += val
        for idx, val in v_indices:
            V22[idx, col] += val
        for idx, val in w_indices:
            W22[idx, col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)

    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    def check_solution(U_mat, V_mat, W_mat, r):
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U_mat[i*m+j, rr] * V_mat[j*p+k, rr] * W_mat[k*n+i, rr]
                    max_err = max(max_err, abs(val - 1))
        return max_err

    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=5000):
        """Try to reduce rank by optimizing with target_rank columns."""
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        
        # Precompute index arrays for efficiency
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        def compute_residuals(U, V, W):
            res = np.zeros(len(indices))
            for idx, (i, j, k) in enumerate(indices):
                val = 0
                for r in range(target_rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                res[idx] = val - 1
            return res
        
        # Gauss-Newton style optimization
        lr = 0.01
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None, float('inf')
            
            # Compute current residual
            res = compute_residuals(U, V, W)
            max_err = np.max(np.abs(res))
            
            if max_err < 1e-12:
                return (U, V, W), max_err
            
            # Compute gradient using finite differences (approximate)
            # For efficiency, use a few random perturbations to estimate direction
            best_update = None
            best_loss = np.sum(res**2)
            
            # Try small perturbations in random directions
            n_perturb = 10
            for _ in range(n_perturb):
                # Random direction
                dU = np.random.randn(*U.shape) * 0.01
                dV = np.random.randn(*V.shape) * 0.01
                dW = np.random.randn(*W.shape) * 0.01
                
                U_new = U + dU
                V_new = V + dV
                W_new = W + dW
                
                res_new = compute_residuals(U_new, V_new, W_new)
                loss_new = np.sum(res_new**2)
                
                if loss_new < best_loss:
                    best_loss = loss_new
                    best_update = (dU, dV, dW)
            
            if best_update is None:
                # Decrease learning rate
                lr *= 0.5
                if lr < 1e-10:
                    break
                continue
            
            U += best_update[0]
            V += best_update[1]
            W += best_update[2]
        
        final_res = compute_residuals(U, V, W)
        max_err = np.max(np.abs(final_res))
        return (U, V, W), max_err

    def try_least_squares(U_init, V_init, W_init, target_rank, max_nfev=5000):
        """Use scipy's least_squares for rank reduction."""
        from scipy.optimize import least_squares
        
        def residuals(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank :]

            U_mat = U_flat.reshape(n*m, target_rank)
            V_mat = V_flat.reshape(m*p, target_rank)
            W_mat = W_flat.reshape(p*n, target_rank)

            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        res.append(val - 1)
            return np.array(res)

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=max_nfev, ftol=1e-14, xtol=1e-14)
            max_err = np.max(np.abs(result.fun))
            U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            return (U_opt, V_opt, W_opt), max_err
        except Exception as e:
            return None, float('inf')

    best_err = float('inf')
    best_result = None

    # Try zeroing out each column in turn
    for skip_col in range(rank22):
        if time.time() > deadline:
            break

        cols = [c for c in range(rank22) if c != skip_col]
        target_rank = len(cols)

        U_init = U22[:, cols]
        V_init = V22[:, cols]
        W_init = W22[:, cols]

        # Method 1: Least squares
        result_ls, err_ls = try_least_squares(U_init, V_init, W_init, target_rank)
        if result_ls is not None and err_ls < best_err:
            best_err = err_ls
            best_result = result_ls
            if err_ls < 1e-10:
                break

    if best_result is not None and best_err < 1e-8:
        U_opt, V_opt, W_opt = best_result

        # Round to nearest half-integer
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2

        err_round = check_solution(U_round, V_round, W_round, W_opt.shape[1])
        if err_round < 1e-10:
            U_out = [[float(U_round[i, r]) for r in range(U_round.shape[1])] for i in range(U_round.shape[0])]
            V_out = [[float(V_round[i, r]) for r in range(V_round.shape[1])] for i in range(V_round.shape[0])]
            W_out = [[float(W_round[i, r]) for r in range(W_round.shape[1])] for i in range(W_round.shape[0])]
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            err_unrounded = check_solution(U_opt, V_opt, W_opt, U_opt.shape[1])
            if err_unrounded < 1e-10:
                U_out = [[float(U_opt[i, r]) for r in range(U_opt.shape[1])] for i in range(U_opt.shape[0])]
                V_out = [[float(V_opt[i, r]) for r in range(V_opt.shape[1])] for i in range(V_opt.shape[0])]
                W_out = [[float(W_opt[i, r]) for r in range(W_opt.shape[1])] for i in range(W_opt.shape[0])]
                return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to rank 22
    U_out = [[float(U22[i, r]) for r in range(rank22)] for i in range(n*m)]
    V_out = [[float(V22[i, r]) for r in range(rank22)] for i in range(m*p)]
    W_out = [[float(W22[i, r]) for r in range(rank22)] for i in range(p*n)]
    return {"U": U_out, "V": V_out, "W": W_out}
