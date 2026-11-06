"""Direct analytical Bermudan put exercise policy via backward induction with
Gauss-Hermite quadrature and exact (unclipped) Gaussian transition."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using analytical backward induction.

    Args:
        training_paths: array of shape (n_paths, n_dates) with simulated spot prices
        contract: dict with contract parameters

    Returns:
        An object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s limit

    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    if n_dates is None or n_dates <= 0:
        n_dates = 12

    dt = T / n_dates

    # Spot grid for the value function (log-spaced for resolution near 0)
    n_grid = 800
    S_min = 0.01 * K
    S_max = 4.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)

    # Gauss-Hermite quadrature
    n_quad = 150
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)

    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)

    # Exact log-shift for each quadrature point: S_next = S * exp(log_shift)
    log_shifts = drift + vol_step * z_norm  # shape (n_quad,)
    exp_shifts = np.exp(log_shifts)         # shape (n_quad,)
    disc = np.exp(-r * dt)

    # Backward induction
    V_next = np.maximum(K - S_grid, 0.0)

    thresholds = np.zeros(n_dates)

    for t_idx in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 3.0:
            for remaining in range(t_idx, -1, -1):
                thresholds[remaining] = K * 0.85
            break

        # Continuation value: disc * E[V_next(S * exp(log_shift))]
        # S_next[j, q] = S_grid[j] * exp_shifts[q]  (exact, no clipping)
        S_next = S_grid[:, np.newaxis] * exp_shifts[np.newaxis, :]
        V_interp = np.interp(S_next.ravel(), S_grid, V_next).reshape(S_next.shape)
        cont_value = disc * (V_interp @ w_norm)

        immediate = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(immediate, cont_value)
        diff = immediate - cont_value

        # Coarse threshold: last grid index where diff >= 0
        pos_indices = np.where(diff >= 0)[0]
        if len(pos_indices) > 0:
            last_pos = pos_indices[-1]
            if last_pos + 1 < n_grid:
                d1 = diff[last_pos]
                d2 = diff[last_pos + 1]
                if d1 >= 0 and d2 < 0:
                    frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                    thresh = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                else:
                    thresh = S_grid[last_pos]
            else:
                thresh = S_grid[last_pos]
        else:
            thresh = 0.0

        # Refine threshold by bisection on the smooth value function
        thresh = np.clip(thresh, 0.0, K)
        if thresh > 0.0:
            lo = 0.0
            hi = thresh
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                mid = 0.5 * (lo + hi)
                # value at a single point mid
                S_m = mid * exp_shifts
                V_m = np.interp(S_m, S_grid, V_next)
                cont_m = disc * np.dot(V_m, w_norm)
                imm_m = max(K - mid, 0.0)
                d_m = imm_m - cont_m
                if d_m >= 0:
                    lo = mid
                else:
                    hi = mid
            thresh = 0.5 * (lo + hi)
            thresh = np.clip(thresh, 0.0, K)

        thresholds[t_idx] = thresh
        V_next = V_current

    # Final high-precision polish: refine each threshold using a fine local grid
    if time.time() < deadline - 10.0:
        for t_idx in range(n_dates):
            if time.time() > deadline - 5.0:
                break
            thresh = thresholds[t_idx]
            if thresh <= 0.0:
                continue
            # Build a fine local grid around the threshold
            local_width = max(0.01 * K, 0.1)
            local_lo = max(0.0, thresh - local_width)
            local_hi = min(K, thresh + local_width)
            if local_hi <= local_lo:
                continue
            n_local = 200
            S_local = np.linspace(local_lo, local_hi, n_local)
            
            # Compute continuation values on local grid
            S_next_local = S_local[:, np.newaxis] * exp_shifts[np.newaxis, :]
            V_interp_local = np.interp(S_next_local.ravel(), S_grid, V_next).reshape(S_next_local.shape)
            cont_local = disc * (V_interp_local @ w_norm)
            imm_local = np.maximum(K - S_local, 0.0)
            diff_local = imm_local - cont_local
            
            # Find sign change
            pos_idx = np.where(diff_local >= 0)[0]
            if len(pos_idx) > 0:
                last_pos = pos_idx[-1]
                if last_pos + 1 < n_local:
                    d1 = diff_local[last_pos]
                    d2 = diff_local[last_pos + 1]
                    if d1 >= 0 and d2 < 0:
                        frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                        thresh_fine = S_local[last_pos] + frac * (S_local[last_pos + 1] - S_local[last_pos])
                    else:
                        thresh_fine = S_local[last_pos]
                else:
                    thresh_fine = S_local[last_pos]
            else:
                thresh_fine = 0.0
            
            # Bisection polish on fine grid
            thresh_fine = np.clip(thresh_fine, local_lo, local_hi)
            if thresh_fine > local_lo:
                lo_f = local_lo
                hi_f = thresh_fine
                for _ in range(30):
                    if time.time() > deadline - 5.0:
                        break
                    mid_f = 0.5 * (lo_f + hi_f)
                    S_m_f = mid_f * exp_shifts
                    V_m_f = np.interp(S_m_f, S_grid, V_next)
                    cont_m_f = disc * np.dot(V_m_f, w_norm)
                    imm_m_f = max(K - mid_f, 0.0)
                    d_m_f = imm_m_f - cont_m_f
                    if d_m_f >= 0:
                        lo_f = mid_f
                    else:
                        hi_f = mid_f
                thresh_fine = 0.5 * (lo_f + hi_f)
                thresh_fine = np.clip(thresh_fine, 0.0, K)
            
            thresholds[t_idx] = thresh_fine

    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time_idx):
            current_state = np.asarray(current_state)
            if time_idx < 0 or time_idx >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return current_state < threshold

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
