import numpy as np
import time
import scipy.optimize as optimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # safety margin before 630s limit
    
    N = 8

    # Precompute quadrature
    c_vals, w_vals = np.polynomial.legendre.leggauss(100)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    Nc = len(c_vals)

    def compute_volume(X, Y):
        """
        Compute the union volume of the 64 tubes with separable product structure.
        X: shape (8,) base x-coords for i-index
        Y: shape (8,) base y-coords for j-index
        At height c, x-intervals for i: [X[i] + c*i/N, X[i] + c*i/N + 1/N]
        y-intervals for j: [Y[j] + c*j/N, Y[j] + c*j/N + 1/N]
        Union area = (1D x-union length) * (1D y-union length) due to product structure.
        """
        # Vectorize over c
        # starts_x shape: (Nc, N)
        idx = np.arange(N)
        starts_x = X[None, :] + c_vals[:, None] * idx[None, :] / N
        ends_x = starts_x + 1.0 / N
        
        starts_y = Y[None, :] + c_vals[:, None] * idx[None, :] / N
        ends_y = starts_y + 1.0 / N
        
        # Sort and merge for each c
        order_x = np.argsort(starts_x, axis=1)
        s_x = np.take_along_axis(starts_x, order_x, axis=1)
        e_x = np.take_along_axis(ends_x, order_x, axis=1)
        
        order_y = np.argsort(starts_y, axis=1)
        s_y = np.take_along_axis(starts_y, order_y, axis=1)
        e_y = np.take_along_axis(ends_y, order_y, axis=1)
        
        # Merge intervals (vectorized over c)
        len_x = np.zeros(Nc)
        len_y = np.zeros(Nc)
        
        for dim, s, e in [(0, s_x, e_x), (1, s_y, e_y)]:
            cur_s = s[:, 0].copy()
            cur_e = e[:, 0].copy()
            length = np.zeros(Nc)
            for k in range(1, N):
                overlap = s[:, k] <= cur_e
                cur_e = np.maximum(cur_e, e[:, k])
                # When no overlap, add current segment
                length += np.where(overlap, 0.0, cur_e - cur_s)
                # Update for next iteration
                new_s = np.where(overlap, cur_s, s[:, k])
                new_e = np.where(overlap, cur_e, e[:, k])
                cur_s = new_s
                cur_e = new_e
            length += cur_e - cur_s
            if dim == 0:
                len_x = length
            else:
                len_y = length
        
        total_vol = np.sum(len_x * len_y * w_vals)
        return total_vol

    def compute_volume_single(X_full):
        """For symmetric case (Y=X), used in initial optimization."""
        return compute_volume(X_full, X_full)

    # Initial guess: linear shearing
    X0 = -np.arange(N) / N

    def obj_sym(X_opt):
        X_full = np.concatenate(([0.0], X_opt))
        return compute_volume_single(X_full)

    # Start with symmetric optimization (7 params), then asymmetric (14 params)
    X_init = X0[1:] - X0[0]
    X_best_sym = np.concatenate(([0.0], X_init))
    best_val = obj_sym(X_init)

    # Phase 1: Optimize symmetric case
    rng = np.random.default_rng(42)
    initializations = [
        X0[1:] - X0[0],  # linear shearing
        np.zeros(N - 1),  # all zero
        rng.uniform(-0.5, 0.5, N - 1),  # random
        -np.arange(1, N) / (2.0 * N),  # negative ramp
        np.arange(1, N) / (2.0 * N),  # positive ramp
        rng.uniform(-1.0, 1.0, N - 1) * 0.3,  # random wider
        rng.uniform(-0.8, 0.8, N - 1) * 0.5,  # another random
    ]

    for init in initializations:
        if time.time() - start_time > timeout - 120:
            break

        try:
            res = optimize.minimize(obj_sym, init, method='Powell',
                                    options={'maxiter': 1500, 'xtol': 1e-9, 'ftol': 1e-12})
            if time.time() - start_time > timeout - 120:
                break
            if res.fun < best_val:
                X_best_sym = np.concatenate(([0.0], res.x))
                best_val = res.fun
        except Exception:
            pass

    # Phase 2: Asymmetric optimization (14 params: X[1..7] and Y[1..7])
    def obj_asym(params):
        X_opt = params[:N-1]
        Y_opt = params[N-1:]
        X_full = np.concatenate(([0.0], X_opt))
        Y_full = np.concatenate(([0.0], Y_opt))
        return compute_volume(X_full, Y_full)

    # Initialize asymmetric from best symmetric
    asym_init = np.concatenate([X_best_sym[1:], X_best_sym[1:]])
    asym_best = asym_init.copy()
    asym_best_val = obj_asym(asym_best)

    if time.time() - start_time < timeout - 60:
        # Try a few asymmetric initializations
        asym_inits = [
            asym_init,
            np.concatenate([X_best_sym[1:], np.zeros(N-1)]),
            np.concatenate([np.zeros(N-1), X_best_sym[1:]]),
            rng.uniform(-0.3, 0.3, 2*(N-1)),
        ]
        
        for ainit in asym_inits:
            if time.time() - start_time > timeout - 30:
                break
            try:
                res_a = optimize.minimize(obj_asym, ainit, method='Powell',
                                          options={'maxiter': 1000, 'xtol': 1e-9, 'ftol': 1e-12})
                if time.time() - start_time > timeout - 30:
                    break
                if res_a.fun < asym_best_val:
                    asym_best = res_a.x.copy()
                    asym_best_val = res_a.fun
            except Exception:
                pass

    # Final polish on asymmetric
    if time.time() - start_time < timeout - 10:
        try:
            res_final = optimize.minimize(obj_asym, asym_best, method='Powell',
                                          options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-13})
            if res_final.fun < asym_best_val:
                asym_best = res_final.x.copy()
                asym_best_val = res_final.fun
        except Exception:
            pass

    # Extract X and Y from best asymmetric solution
    X_final = np.concatenate(([0.0], asym_best[:N-1]))
    Y_final = np.concatenate(([0.0], asym_best[N-1:]))

    # Construct positions
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_final[i]
        positions[k, 1] = Y_final[j]

    return positions
