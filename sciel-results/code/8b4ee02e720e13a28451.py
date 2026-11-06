"""Bermudan put exercise policy via analytical backward induction.

Uses a log-spaced value-function grid, exact (unclipped) Gaussian transition
evaluated with Gauss-Hermite quadrature, and direct root-finding on the smooth
immediate-minus-continuation function to locate each exercise threshold to
machine precision. A final local high-resolution polish refines each threshold.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cont_at(S, S_grid, V_next, exp_shifts, w_norm, disc):
    """Continuation value at scalar S using the stored next-step value function."""
    S_next = S * exp_shifts
    V_interp = np.interp(S_next, S_grid, V_next)
    return disc * np.dot(V_interp, w_norm)


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
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    if n_dates is None or n_dates <= 0:
        n_dates = 12

    dt = T / n_dates

    # Value-function grid (log-spaced for resolution near 0)
    n_grid = 2000
    S_min = 0.005 * K
    S_max = 5.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)

    # Gauss-Hermite quadrature (exact Gaussian transition)
    n_quad = 260
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)

    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    exp_shifts = np.exp(drift + vol_step * z_norm)  # shape (n_quad,)
    disc = np.exp(-r * dt)

    # Backward induction
    V_next = np.maximum(K - S_grid, 0.0)

    thresholds = np.zeros(n_dates)

    def f_at(S):
        """Smooth immediate-minus-continuation at scalar S."""
        imm = K - S
        if imm < 0.0:
            imm = 0.0
        cont = _compute_cont_at(S, S_grid, V_next, exp_shifts, w_norm, disc)
        return imm - cont

    for t_idx in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 8.0:
            # Fallback heuristic for remaining dates
            for remaining in range(t_idx, -1, -1):
                thresholds[remaining] = K * 0.85
            break

        # Coarse scan to bracket the zero of f (f decreasing in S)
        scan = np.geomspace(S_min, min(K, S_max), 400)
        f_scan = np.empty_like(scan)
        for i in range(len(scan)):
            f_scan[i] = f_at(scan[i])

        # Find last index where f >= 0 (exercise region is S < threshold)
        pos_idx = np.where(f_scan >= 0.0)[0]
        if len(pos_idx) == 0:
            thresh = 0.0
        else:
            last_pos = pos_idx[-1]
            if last_pos + 1 < len(scan) and f_scan[last_pos + 1] < 0.0:
                # Linear interp to bracket
                d1 = f_scan[last_pos]
                d2 = f_scan[last_pos + 1]
                frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                thresh0 = scan[last_pos] + frac * (scan[last_pos + 1] - scan[last_pos])
                lo = scan[last_pos]
                hi = scan[last_pos + 1]
            else:
                # f positive everywhere scanned -> threshold beyond scan
                lo = scan[-1]
                hi = min(K, S_max)
                thresh0 = hi
            # Bisection on the smooth f to machine precision
            lo = max(0.0, lo)
            hi = min(K, hi)
            if hi > lo:
                for _ in range(70):
                    if time.time() > deadline - 8.0:
                        break
                    mid = 0.5 * (lo + hi)
                    fm = f_at(mid)
                    if fm >= 0.0:
                        lo = mid
                    else:
                        hi = mid
                thresh = 0.5 * (lo + hi)
            else:
                thresh = lo

        thresh = np.clip(thresh, 0.0, K)
        thresholds[t_idx] = thresh

        # Update value function for next step back
        S_next_grid = S_grid[:, np.newaxis] * exp_shifts[np.newaxis, :]
        V_interp = np.interp(S_next_grid.ravel(), S_grid, V_next).reshape(S_next_grid.shape)
        cont_value = disc * (V_interp @ w_norm)
        immediate = np.maximum(K - S_grid, 0.0)
        V_next = np.maximum(immediate, cont_value)

    # Final high-resolution local polish on each threshold
    if time.time() < deadline - 10.0:
        for t_idx in range(n_dates):
            if time.time() > deadline - 5.0:
                break
            thresh = thresholds[t_idx]
            if thresh <= 0.0:
                continue
            local_width = max(0.005 * K, 0.05)
            local_lo = max(0.0, thresh - local_width)
            local_hi = min(K, thresh + local_width)
            if local_hi <= local_lo:
                continue
            n_local = 600
            S_local = np.linspace(local_lo, local_hi, n_local)
            S_next_local = S_local[:, np.newaxis] * exp_shifts[np.newaxis, :]
            V_interp_local = np.interp(S_next_local.ravel(), S_grid, V_next).reshape(S_next_local.shape)
            cont_local = disc * (V_interp_local @ w_norm)
            imm_local = np.maximum(K - S_local, 0.0)
            diff_local = imm_local - cont_local

            pos_idx = np.where(diff_local >= 0.0)[0]
            if len(pos_idx) > 0:
                last_pos = pos_idx[-1]
                if last_pos + 1 < n_local and diff_local[last_pos + 1] < 0.0:
                    d1 = diff_local[last_pos]
                    d2 = diff_local[last_pos + 1]
                    frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                    thresh_fine = S_local[last_pos] + frac * (S_local[last_pos + 1] - S_local[last_pos])
                    lo_f = S_local[last_pos]
                    hi_f = S_local[last_pos + 1]
                else:
                    thresh_fine = S_local[last_pos]
                    lo_f = max(0.0, thresh - local_width)
                    hi_f = S_local[last_pos]
            else:
                thresh_fine = local_lo
                lo_f = local_lo
                hi_f = local_lo

            thresh_fine = np.clip(thresh_fine, local_lo, local_hi)
            if hi_f > lo_f:
                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    mid_f = 0.5 * (lo_f + hi_f)
                    fm_f = f_at(mid_f)
                    if fm_f >= 0.0:
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
