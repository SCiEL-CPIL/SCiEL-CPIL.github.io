import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _compute_P(support, weights):
    """Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0

    # Pairwise sums S_ij = s_i + s_j, shape (n, n)
    S = s[:, None] + s[None, :]
    # R_kl = 2*s_l - s_k, shape (n, n)
    R = 2 * s[None, :] - s[:, None]

    # Full 4D comparison: S[i,j] < R[k,l]
    mask = S[:, :, None, None] < R[None, None, :, :]

    # Weight product: w_i * w_j * w_k * w_l
    W_prod = (w[:, None, None, None] *
              w[None, :, None, None] *
              w[None, None, :, None] *
              w[None, None, None, :])

    return float(np.sum(W_prod * mask))


def _neg_P(params, n):
    """Negative P for minimization. params = [s_0,...,s_{n-1}, log_w_0,...,log_w_{n-1}]"""
    s = np.exp(params[:n])
    lw = params[n:]
    w = np.exp(lw)
    w = w / w.sum()
    return -_compute_P(s, w)


def _try_optimize_n(n, start_time, deadline, maxiter=500, tol=1e-10):
    """Optimize over continuous support and weights for given n."""
    if time.time() > deadline - 30:
        return None, -np.inf

    # Initial guess: evenly spaced support, uniform weights
    s0 = np.linspace(0.1, 2.0, n)
    w0 = np.ones(n) / n
    # Log-weights for unconstrained optimization
    lw0 = np.log(w0)

    # Bounds: support in [0.01, 100], log-weights in [-20, 0] (weights in [~0, 1])
    bounds = [(0.01, 100.0)] * n + [(-20.0, 0.0)] * n

    # Differential evolution for global search
    try:
        result_de = differential_evolution(
            _neg_P, bounds, args=(n,),
            maxiter=maxiter, tol=1e-8, seed=42,
            popsize=15, mutation=(0.3, 1.0), recombination=0.7
        )
    except Exception:
        return None, -np.inf

    best_params = result_de.x
    best_P_val = -result_de.fun

    # Nelder-Mead polish
    if time.time() < deadline - 15:
        try:
            result_nm = minimize(
                _neg_P, best_params, args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result_nm.fun < result_de.fun:
                best_params = result_nm.x
                best_P_val = -result_nm.fun
        except Exception:
            pass

    if best_P_val <= -np.inf:
        return None, -np.inf

    s_opt = np.exp(best_params[:n])
    lw_opt = best_params[n:]
    w_opt = np.exp(lw_opt)
    w_opt = w_opt / w_opt.sum()

    return (s_opt, w_opt), best_P_val


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s kill

    # 1. Seed solution
    n_seed = 32
    support_seed = np.arange(n_seed, dtype=np.float64)
    weights_seed = np.ones(n_seed, dtype=np.float64) / n_seed
    best_P = _compute_P(support_seed, weights_seed)
    best_support = support_seed.copy()
    best_weights = weights_seed.copy()

    # 2. Optimized Binary
    p_opt = (1 + np.sqrt(33)) / 16
    support_bin = np.array([0.0, 1.0])
    weights_bin = np.array([1 - p_opt, p_opt])
    P_bin = _compute_P(support_bin, weights_bin)
    if P_bin > best_P:
        best_P = P_bin
        best_support = support_bin.copy()
        best_weights = weights_bin.copy()

    # 3. Optimize continuous support and weights for n=3,4,5
    for n in [3, 4, 5]:
        if time.time() > deadline - 60:
            break

        # Try multiple initializations with different support scales
        for scale in [1.0, 2.0, 5.0, 10.0]:
            if time.time() > deadline - 30:
                break

            # Temporarily modify bounds by scaling initial guess
            # We'll just run the optimizer; it should explore the space
            res, P_val = _try_optimize_n(n, start_time, deadline)
            if res is not None and P_val > best_P:
                best_P = P_val
                best_support = res[0].copy()
                best_weights = res[1].copy()

    # 4. Local refinement of best solution if it's small-n
    if len(best_support) <= 5 and time.time() < deadline - 20:
        n = len(best_support)
        # Re-parameterize current best into log-space
        s_cur = best_support.copy()
        w_cur = best_weights.copy()
        # Ensure all positive
        s_cur = np.maximum(s_cur, 0.01)
        w_cur = np.maximum(w_cur, 1e-15)
        w_cur = w_cur / w_cur.sum()

        params_cur = np.concatenate([np.log(s_cur), np.log(w_cur)])

        # Multiple Nelder-Mead restarts with small perturbations
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            if trial == 0:
                p_init = params_cur.copy()
            else:
                p_init = params_cur + np.random.randn(len(params_cur)) * 0.1

            try:
                result = minimize(
                    _neg_P, p_init, args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                P_trial = -result.fun
                if P_trial > best_P:
                    s_trial = np.exp(result.x[:n])
                    lw_trial = result.x[n:]
                    w_trial = np.exp(lw_trial)
                    w_trial = w_trial / w_trial.sum()
                    best_P = P_trial
                    best_support = s_trial
                    best_weights = w_trial
                    params_cur = result.x.copy()
            except Exception:
                pass

    # Final sanity check
    if (np.any(best_weights < -1e-9) or np.any(best_support < -1e-9) or
            np.sum(best_weights) <= 1e-9 or
            np.any(np.isnan(best_weights)) or np.any(np.isnan(best_support)) or
            np.any(np.isinf(best_weights)) or np.any(np.isinf(best_support))):
        return support_seed, weights_seed

    return best_support, best_weights
