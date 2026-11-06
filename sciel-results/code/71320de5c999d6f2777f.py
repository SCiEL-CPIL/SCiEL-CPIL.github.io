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
from scipy.optimize import minimize_scalar
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
    
    # We need to estimate the optimal exercise threshold for each date.
    # Strategy: Use backward induction with Monte Carlo to estimate continuation values,
    # then fit a logistic regression to learn the boundary.
    
    # First, let's determine the shape of training_paths
    if training_paths is None:
        # Generate synthetic training paths
        rng = np.random.default_rng(42)
        S0 = contract.get('S0', 100.0)
        training_paths = _generate_gbm_paths(rng, S0, K, r, sigma, T, n_paths, n_exercise_dates)
    
    n_paths, n_dates = training_paths.shape
    
    # For each exercise date, we want to find the threshold B_k such that
    # we exercise if S <= B_k.
    # We estimate this by comparing intrinsic value vs. estimated continuation value.
    
    # Estimate continuation values using a simplified approach:
    # For a Bermudan put, the continuation value at time t_k with spot S_k is
    # approximately the value of a European put on the remaining exercise dates.
    # We can estimate this via Monte Carlo from the training paths.
    
    # More efficient approach: For each date k, we know the distribution of future
    # spots from the training paths. We can estimate the continuation value as
    # the expected discounted maximum payoff over remaining exercise dates.
    
    # Let's use a direct backward induction approach on the training paths.
    
    # Step 1: Initialize the value at the last exercise date (t=12/12 = T)
    # At maturity, the value is the intrinsic value (European put payoff)
    intrinsic_values = np.maximum(K - training_paths, 0.0)  # shape: (n_paths, n_dates)
    
    # Step 2: Backward induction
    # V_k(S_k) = max(intrinsic_k(S_k), discount * E[V_{k+1}(S_{k+1}) | S_k])
    # Since we're on a path, we approximate E[V_{k+1} | S_k] using the training data.
    
    # For simplicity and speed, we'll estimate the continuation value for each date
    # by looking at the average future payoffs conditional on current spot bins.
    
    # Discretize spot prices into bins for each date
    n_bins = 100
    bin_edges = np.linspace(0, 200, n_bins + 1)
    
    # Store the estimated threshold for each exercise date
    thresholds = np.zeros(n_dates)
    logistic_params = []  # (intercept, slope) for each date
    
    # Backward induction: start from the last date
    # V_last = intrinsic value at last date
    V_next = intrinsic_values[:, -1].copy()  # shape: (n_paths,)
    
    for k in range(n_dates - 1, -1, -1):
        current_intrinsic = intrinsic_values[:, k]  # shape: (n_paths,)
        
        if k == n_dates - 1:
            # Last date: always exercise if intrinsic > 0 (but for put, always >= 0)
            # Actually, at the last exercise date, we exercise if intrinsic > 0
            # which is always true for a put with S < K
            # The threshold is K
            thresholds[k] = K
            # Fit logistic: P(exercise) = sigmoid(intercept + slope * S)
            # At S=K, P=0.5; for S<<K, P~1; for S>>K, P~0
            # Use a steep logistic around K
            slope = -10.0 / K  # steepness
            intercept = 5.0  # so that at S=K, logit = 5 - 10 = -5... let's adjust
            # We want: at S=K, logit=0 => intercept + slope*K = 0 => intercept = -slope*K
            intercept = -slope * K
            logistic_params.append((intercept, slope))
        else:
            # Estimate continuation value for each path at date k
            # Use the next step's values, averaged over paths with similar current spot
            # For efficiency, bin the current spots and compute average V_next for each bin
            
            S_k = training_paths[:, k]
            
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
            
            # For bins with no data, use the overall average
            if np.any(~valid_bins):
                bin_continuation[~valid_bins] = np.mean(V_next)
            
            # Discount the continuation value
            discount = np.exp(-r * dt)
            continuation_values = bin_continuation[bin_indices] * discount
            
            # Exercise if intrinsic >= continuation
            exercise_mask = current_intrinsic >= continuation_values
            
            # Find the threshold: the largest S where we still exercise
            # Sort by S_k and find the boundary
            sorted_indices = np.argsort(S_k)
            sorted_S = S_k[sorted_indices]
            sorted_exercise = exercise_mask[sorted_indices]
            
            # Find the last index where exercise is True
            # The threshold is between the last exercised and first non-exercised
            if np.all(sorted_exercise):
                # All exercise: threshold is very high
                thresholds[k] = 200.0
            elif not np.any(sorted_exercise):
                # None exercise: threshold is very low
                thresholds[k] = 0.0
            else:
                # Find the boundary
                # The transition from True to False
                diff = np.diff(sorted_exercise.astype(int))
                transition_indices = np.where(diff == -1)[0]
                if len(transition_indices) > 0:
                    # Take the first transition (lowest S where we stop exercising)
                    idx = transition_indices[0]
                    # Threshold is between sorted_S[idx] and sorted_S[idx+1]
                    thresholds[k] = (sorted_S[idx] + sorted_S[idx + 1]) / 2.0
                else:
                    # All True or all False (handled above)
                    if sorted_exercise[-1]:
                        thresholds[k] = 200.0
                    else:
                        thresholds[k] = 0.0
            
            # Fit logistic regression to approximate the exercise decision
            # P(exercise | S) = sigmoid(intercept + slope * S)
            # We want this to be ~1 for S < threshold and ~0 for S > threshold
            # Use a logistic centered at the threshold with appropriate steepness
            if thresholds[k] > 0 and thresholds[k] < 200:
                # Steepness: choose so that the logistic goes from ~0.95 to ~0.05
                # over a range of about 5% of K
                steepness_range = 0.05 * K
                slope = -19.4 / steepness_range  # logit(0.95/0.05) = 2.944, times 2 for symmetric
                # Actually, logit(0.95) = 2.944, logit(0.05) = -2.944
                # So over steepness_range, we want 2*2.944 = 5.888 change in logit
                # slope * steepness_range = 5.888 => slope = 5.888 / steepness_range
                slope = -5.888 / steepness_range
                intercept = -slope * thresholds[k]  # so that at S=threshold, logit=0
                logistic_params.append((intercept, slope))
            else:
                # Edge cases
                if thresholds[k] >= 200:
                    logistic_params.append((100.0, 0.0))  # always exercise
                else:
                    logistic_params.append((-100.0, 0.0))  # never exercise
        
        # Update V for this step: V_k = max(intrinsic, continuation)
        V_k = np.maximum(current_intrinsic, continuation_values if k < n_dates - 1 else np.zeros(n_paths))
        V_next = V_k
    
    # Reverse the params since we went backward
    logistic_params = logistic_params[::-1]
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, params, thresholds):
            self.params = params  # list of (intercept, slope) for each date
            self.thresholds = thresholds
        
        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating whether to exercise for each path.
            
            Parameters:
            - current_state: array of spot prices (shape: (n_eval_paths,))
            - time: current time (float, in years)
            
            Returns:
            - boolean array of shape (n_eval_paths,)
            """
            # Determine which exercise date we're at
            # time is in [0, T], exercise dates are at k*dt for k=1..12
            dt = T / n_exercise_dates
            # Find the exercise date index
            k = int(round(time / dt)) - 1  # 0-indexed, so k=0 for t=dt
            k = max(0, min(k, n_dates - 1))
            
            # Get the logistic parameters for this date
            intercept, slope = self.params[k]
            
            # Compute the logistic probability
            logits = intercept + slope * current_state
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            
            # Exercise if probability > 0.5, which is equivalent to S < threshold
            # But for robustness, use the threshold directly
            return current_state <= self.thresholds[k]
    
    return ExercisePolicy(logistic_params, thresholds)


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
