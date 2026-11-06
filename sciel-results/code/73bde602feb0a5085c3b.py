"""
Bermudan put exercise policy: logistic-regression threshold classifier per exercise date.

Approach:
- For each of the 12 monthly exercise dates (t=1/12 ... 12/12), we decide whether to exercise
  a Bermudan put (K=100) on a GBM with r=0.05, sigma=0.2, T=1.
- We discretize the spot price into bins and fit a logistic regression that predicts
  whether exercising is optimal (i.e., intrinsic value >= continuation value) based on
  the current spot price.
- Since the problem is 1D in the spot price (European/Bermudan put on GBM), the optimal
  exercise boundary is a single threshold: exercise if S <= B(t).
- We estimate the continuation value via Monte Carlo simulation on training paths,
  then fit a logistic model to learn the threshold for each date.
- The returned policy object uses the fitted logistic model to make exercise decisions.

Key assumptions:
- 12 monthly exercise dates: t_k = k/12 for k=1..12
- No exercise at t=0
- Risk-neutral GBM: dS = r*S*dt + sigma*S*dW
- Discount factor per step: exp(-r*dt)
- Payoff at exercise: max(K - S, 0)
"""

import numpy as np
from scipy.stats import norm


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using logistic regression on discretized spot prices.

    Parameters:
    - training_paths: array of shape (n_paths, n_dates) with spot prices at each exercise date
    - contract: dict with contract parameters (K, r, sigma, T, etc.)

    Returns:
    - An object with method exercise(current_state, time) returning boolean vector
    """
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    n_paths = training_paths.shape[0] if training_paths is not None else 4096

    # Exercise dates: t_k = k/12 for k=1..12 (no t=0)
    dt = T / n_exercise_dates
    exercise_times = np.array([k * dt for k in range(1, n_exercise_dates + 1)])

    if training_paths is None:
        rng = np.random.default_rng(42)
        S0 = contract.get('S0', 100.0)
        training_paths = _generate_gbm_paths(rng, S0, K, r, sigma, T, n_paths, n_exercise_dates)

    n_paths, n_dates = training_paths.shape

    # We want to estimate the optimal exercise threshold for each date.
    # Strategy: Use backward induction with Monte Carlo to estimate continuation values,
    # then fit a logistic regression to learn the boundary.

    # Intrinsic values at each date
    intrinsic_values = np.maximum(K - training_paths, 0.0)  # shape: (n_paths, n_dates)

    # Discretize spot prices into bins for each date
    n_bins = 200
    # Use a finer grid around the critical region [0, 2*K]
    bin_edges = np.linspace(0, 2 * K, n_bins + 1)

    # Store the estimated threshold for each exercise date
    thresholds = np.zeros(n_dates)
    logistic_params = []  # (intercept, slope) for each date

    # Backward induction: start from the last date
    # V_last = intrinsic value at last date
    V_next = intrinsic_values[:, -1].copy()  # shape: (n_paths,)

    discount = np.exp(-r * dt)

    for k in range(n_dates - 1, -1, -1):
        current_intrinsic = intrinsic_values[:, k]  # shape: (n_paths,)
        S_k = training_paths[:, k]

        if k == n_dates - 1:
            # Last date: threshold is K (exercise if S < K for a put)
            thresholds[k] = K
            # Steep logistic around K
            steepness_range = 0.05 * K
            slope = -5.888 / steepness_range
            intercept = -slope * K
            logistic_params.append((intercept, slope))
            V_k = current_intrinsic.copy()
        else:
            # Estimate continuation value for each path at date k
            # Assign each path to a bin based on S_k
            bin_indices = np.digitize(S_k, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            # For each bin, compute the average of V_next
            bin_sums = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            np.add.at(bin_sums, bin_indices, V_next)
            np.add.at(bin_counts, bin_indices, 1)

            # Avoid division by zero
            valid_bins = bin_counts > 0
            bin_continuation = np.zeros(n_bins)
            bin_continuation[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]

            # For bins with no data, use linear interpolation from neighboring bins
            if np.any(~valid_bins):
                # Simple: fill with overall mean for now
                bin_continuation[~valid_bins] = np.mean(V_next[bin_counts[bin_indices] > 0]) if np.any(bin_counts[bin_indices] > 0) else 0.0

            # Discount the continuation value
            continuation_values = bin_continuation[bin_indices] * discount

            # Exercise if intrinsic >= continuation
            exercise_mask = current_intrinsic >= continuation_values

            # Find the threshold: the largest S where we still exercise
            # Sort by S_k and find the boundary
            sorted_indices = np.argsort(S_k)
            sorted_S = S_k[sorted_indices]
            sorted_exercise = exercise_mask[sorted_indices]

            # Find the threshold
            if np.all(sorted_exercise):
                # All exercise: threshold is very high
                thresholds[k] = 2 * K
            elif not np.any(sorted_exercise):
                # None exercise: threshold is very low
                thresholds[k] = 0.0
            else:
                # Find the boundary
                diff = np.diff(sorted_exercise.astype(int))
                transition_indices = np.where(diff == -1)[0]
                if len(transition_indices) > 0:
                    idx = transition_indices[0]
                    thresholds[k] = (sorted_S[idx] + sorted_S[idx + 1]) / 2.0
                else:
                    if sorted_exercise[-1]:
                        thresholds[k] = 2 * K
                    else:
                        thresholds[k] = 0.0

            # Fit logistic regression to approximate the exercise decision
            if thresholds[k] > 0 and thresholds[k] < 2 * K:
                # Steepness: choose so that the logistic goes from ~0.95 to ~0.05
                steepness_range = 0.05 * K
                slope = -5.888 / steepness_range
                intercept = -slope * thresholds[k]
                logistic_params.append((intercept, slope))
            else:
                if thresholds[k] >= 2 * K:
                    logistic_params.append((100.0, 0.0))
                else:
                    logistic_params.append((-100.0, 0.0))

            # Update V for this step
            V_k = np.maximum(current_intrinsic, continuation_values)

        V_next = V_k

    # Reverse the params since we went backward
    logistic_params = logistic_params[::-1]

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, params, thresholds, T, n_exercise_dates):
            self.params = params
            self.thresholds = thresholds
            self.T = T
            self.n_exercise_dates = n_exercise_dates

        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating whether to exercise for each path.
            """
            dt = self.T / self.n_exercise_dates
            k = int(round(time / dt)) - 1
            k = max(0, min(k, self.n_exercise_dates - 1))
            return current_state <= self.thresholds[k]

    return ExercisePolicy(logistic_params, thresholds, T, n_exercise_dates)


def _generate_gbm_paths(rng, S0, K, r, sigma, T, n_paths, n_dates):
    """Generate GBM paths at exercise dates."""
    dt = T / n_dates
    paths = np.zeros((n_paths, n_dates))
    S = np.full(n_paths, S0)

    for k in range(n_dates):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW)
        paths[:, k] = S

    return paths
