import numpy as np
import scipy.optimize as optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # safety margin before 630s limit

    N = 8

    # Precompute quadrature points outside the function
    c_vals, w_vals = np.polynomial.legendre.leggauss(100)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals

    def compute_volume(X):
        """
        Compute the union volume of the 64 tubes.
        X is an array of shape (8,) representing the base x-coordinates for each i.
        The y-coordinates are the same as x-coordinates (symmetry).
        """

        total_vol = 0.0
        for idx in range(len(c_vals)):
            c = c_vals[idx]
            w = w_vals[idx]
            # Intervals for each i
            starts = X + c * np.arange(N) / N
            ends = starts + 1.0 / N

            # Sort by start
            order = np.argsort(starts)
            s = starts[order]
            e = ends[order]

            # Merge intervals
            cur_s = s[0]
            cur_e = e[0]
            length = 0.0
            for k in range(1, N):
                if s[k] <= cur_e:
                    cur_e = max(cur_e, e[k])
                else:
                    length += cur_e - cur_s
                    cur_s = s[k]
                    cur_e = e[k]
            length += cur_e - cur_s

            total_vol += length * length * w

        return total_vol

    # Initial guess: linear shearing X_i = -i/8
    X0 = -np.arange(N) / N

    def obj(X_opt):
        X_full = np.concatenate(([0.0], X_opt))
        return compute_volume(X_full)

    X_init = X0[1:] - X0[0]

    X_best = np.concatenate(([0.0], X_init))
    best_val = obj(X_init)

    # Try multiple optimization strategies with different initializations
    rng = np.random.default_rng(42)
    initializations = [
        X0[1:] - X0[0],  # linear shearing
        np.zeros(N - 1),  # all zero
        rng.uniform(-0.5, 0.5, N - 1),  # random
        -np.arange(1, N) / (2.0 * N),  # negative ramp
        np.arange(1, N) / (2.0 * N),  # positive ramp
        rng.uniform(-1.0, 1.0, N - 1) * 0.3,  # random wider
    ]

    for init in initializations:
        if time.time() - start_time > timeout:
            break

        # Powell's method
        try:
            res = optimize.minimize(obj, init, method='Powell',
                                    options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-11})
            if time.time() - start_time > timeout:
                break
            if res.fun < best_val:
                X_best = np.concatenate(([0.0], res.x))
                best_val = res.fun
        except Exception:
            pass

        # Nelder-Mead from the best point
        if time.time() - start_time > timeout:
            break
        try:
            res2 = optimize.minimize(obj, X_best[1:], method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12})
            if time.time() - start_time > timeout:
                break
            if res2.fun < best_val:
                X_best = np.concatenate(([0.0], res2.x))
                best_val = res2.fun
        except Exception:
            pass

    # Final high-precision polish with tighter tolerance
    if time.time() - start_time < timeout - 30:
        try:
            res3 = optimize.minimize(obj, X_best[1:], method='Powell',
                                     options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-13})
            if time.time() - start_time > timeout:
                pass
            elif res3.fun < best_val:
                X_best = np.concatenate(([0.0], res3.x))
                best_val = res3.fun
        except Exception:
            pass

    # Construct positions
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_best[i]
        positions[k, 1] = X_best[j]

    return positions
