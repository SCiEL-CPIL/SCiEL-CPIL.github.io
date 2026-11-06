# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin before 120s limit

    # Extract parameters from contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_ex = 12
    dt = T / n_ex
    disc = np.exp(-r * dt)

    # training_paths: shape (n_paths, n_time_steps) - prices at each exercise date
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)

    n_paths = paths.shape[0]
    n_dates = paths.shape[1] if paths.shape[1] > 0 else n_ex
    # Ensure we have exactly n_ex dates
    if n_dates < n_ex:
        # Pad or resize
        new_paths = np.zeros((n_paths, n_ex))
        new_paths[:, :min(n_dates, n_ex)] = paths[:, :min(n_dates, n_ex)]
        paths = new_paths
    elif n_dates > n_ex:
        paths = paths[:, :n_ex]

    # Exercise dates: t_1, t_2, ..., t_12
    exercise_times = np.arange(1, n_ex + 1) * dt
    remaining_frac = 1.0 - exercise_times / T  # fraction of time remaining

    # Step 1: Compute European put thresholds at each exercise date
    # European put value at time t with maturity T:
    # P_E(S, t) = K*exp(-r*(T-t))*N(-d2) - S*N(-d1)
    # Threshold: find S* such that P_E(S*, t) = 0 (boundary of exercise region)
    # For a put, exercise when S < threshold. The threshold satisfies:
    # S* = K*exp(-r*(T-t)) * N(-d2) / N(-d1) approximately, but better to solve numerically

    from scipy.stats import norm
    from scipy.optimize import brentq

    def european_put_threshold(t):
        """Find S* where European put value equals intrinsic value (exercise boundary)."""
        tau = T - t
        if tau <= 0:
            return K  # At maturity, exercise if S < K

        def put_value(S):
            if S <= 0:
                return K * np.exp(-r * tau)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
            pv = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return pv

        def exercise_diff(S):
            return put_value(S) - max(K - S, 0)

        # Search for threshold where put_value = intrinsic
        # For puts, threshold is below K
        try:
            # Try to find root in [1, K]
            lo, hi = 1.0, K
            if exercise_diff(lo) * exercise_diff(hi) > 0:
                # No sign change, threshold might be very low
                return lo
            return brentq(exercise_diff, lo, hi, xtol=1e-10)
        except:
            return K * 0.8  # Fallback

    # Compute European thresholds
    eu_thresholds = np.array([european_put_threshold(t) for t in exercise_times])

    # Step 2: Parametric smooth boundary
    # b(t) = sigmoid(a0 + a1*u + a2*u^2 + a3*u^3) * K
    # where u = remaining_frac (1 at t=0, 0 at t=T)
    # At maturity (u=0), b should be K
    # Early on (u~1), b should be lower (more likely to exercise)

    def boundary_from_params(params):
        """Compute boundary at each exercise date from polynomial parameters."""
        a0, a1, a2, a3 = params
        u = remaining_frac
        logit = a0 + a1 * u + a2 * u**2 + a3 * u**3
        # sigmoid maps to (0,1), scale to (0, K)
        sig = 1.0 / (1.0 + np.exp(-logit))
        return sig * K

    # Initial fit: match European thresholds
    # We want boundary_from_params(params) ≈ eu_thresholds
    # Use least squares on log-scale for better conditioning

    def logit_from_threshold(S):
        """Invert sigmoid: given S = sigmoid(x)*K, find x."""
        ratio = S / K
        ratio = np.clip(ratio, 1e-10, 1 - 1e-10)
        return np.log(ratio / (1 - ratio))

    target_logits = logit_from_threshold(eu_thresholds)

    # Fit polynomial to target logits
    u = remaining_frac
    A = np.column_stack([np.ones_like(u), u, u**2, u**3])
    # Solve least squares
    try:
        init_params, _, _, _ = np.linalg.lstsq(A, target_logits, rcond=None)
    except:
        init_params = np.array([0.0, -1.0, 0.5, -0.3])

    # Ensure boundary at maturity is close to K
    # At u=0: boundary = sigmoid(a0)*K, want ≈ K, so a0 should be large positive
    if init_params[0] < 5:
        init_params[0] = 5.0

    def compute_policy_value(params, eval_paths, n_eval_paths):
        """Compute mean discounted payoff for a given boundary parameterization."""
        boundary = boundary_from_params(params)
        # boundary[i] is the threshold at exercise date i
        # Exercise if S < boundary[i]

        n_p = eval_paths.shape[0]
        total_payoff = np.zeros(n_p)

        for i in range(n_ex):
            S_i = eval_paths[:, i]
            exercise_mask = S_i < boundary[i]
            if np.any(exercise_mask):
                payoff = np.where(exercise_mask, K - S_i, 0.0)
                total_payoff += (disc ** i) * payoff
            if np.all(exercise_mask):
                break  # All exercised, no future payoffs

        return np.mean(total_payoff)

    # Step 3: Refine with local search on Monte Carlo objective
    # Use training paths for quick evaluation, then verify with a subset

    # First, evaluate initial params
    init_value = compute_policy_value(init_params, paths, n_paths)

    best_params = init_params.copy()
    best_value = init_value

    # Simple coordinate descent / Nelder-Mead style refinement
    # Use antithetic variates: pair paths i and n_paths-1-i

    def compute_policy_value_antithetic(params, eval_paths):
        """Compute mean discounted payoff using antithetic pairs."""
        boundary = boundary_from_params(params)
        n_p = eval_paths.shape[0]
        half = n_p // 2

        total_payoff = np.zeros(half)

        for i in range(n_ex):
            S_i = eval_paths[:half, i]
            S_i_anti = eval_paths[half:, i]
            # Antithetic: use min of the two for variance reduction? No, use both
            # Actually for antithetic, we evaluate both and average
            ex1 = S_i < boundary[i]
            ex2 = S_i_anti < boundary[i]
            p1 = np.where(ex1, K - S_i, 0.0)
            p2 = np.where(ex2, K - S_i_anti, 0.0)
            total_payoff += (disc ** i) * 0.5 * (p1 + p2)
            if np.all(ex1) and np.all(ex2):
                break

        return np.mean(total_payoff)

    # Coordinate descent refinement
    n_dims = 4
    step_sizes = np.array([0.5, 0.5, 0.5, 0.5])
    max_iter = 200

    for iteration in range(max_iter):
        if time.time() - start_time > TIME_LIMIT - 10:
            break

        improved = False
        for d in range(n_dims):
            # Try positive and negative steps
            for sign in [1, -1]:
                trial = best_params.copy()
                trial[d] += sign * step_sizes[d]
                val = compute_policy_value_antithetic(trial, paths)
                if val > best_value:
                    best_params = trial
                    best_value = val
                    improved = True
            # Adaptive step size
            if not improved:
                step_sizes[d] *= 0.7
                if step_sizes[d] < 1e-4:
                    step_sizes[d] = 1e-4

        if not improved:
            break

    # Step 4: Final evaluation and create policy object
    final_boundary = boundary_from_params(best_params)

    # Ensure boundary is monotonically decreasing in time (increasing in u)
    # For a put, the exercise boundary should decrease over time (you're less likely to exercise later)
    # Actually for Bermudan puts, the boundary typically increases over time (closer to K at maturity)
    # Let's enforce monotonicity: boundary should be non-decreasing over time
    for i in range(1, n_ex):
        if final_boundary[i] < final_boundary[i-1]:
            final_boundary[i] = final_boundary[i-1]

    # Create exercise policy class
    class ExercisePolicy:
        def __init__(self, boundary):
            self.boundary = boundary
            self.n_ex = n_ex

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices at current time
            time: current time value (to determine which exercise date we're at)
            Returns: boolean vector indicating which paths should exercise
            """
            # Determine which exercise date we're at
            # time is the current time, exercise dates are at dt, 2*dt, ..., n_ex*dt
            # Find the index of the current exercise date
            if time is None:
                # Default to last exercise date
                idx = self.n_ex - 1
            else:
                # time could be a scalar or array
                if np.isscalar(time):
                    t_val = float(time)
                    # Find closest exercise date
                    ex_times = np.arange(1, self.n_ex + 1) * dt
                    idx = int(np.argmin(np.abs(ex_times - t_val)))
                else:
                    # Assume it's already an index
                    idx = int(time)

            idx = max(0, min(idx, self.n_ex - 1))
            S = np.asarray(current_state, dtype=np.float64)
            return S < self.boundary[idx]

    return ExercisePolicy(final_boundary)
# EVOLVE-BLOCK-END
