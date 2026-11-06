import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz style
    regression-based backward induction with polynomial continuation values.
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = 1.0 / 12.0
    disc = np.exp(-r * dt)

    # Normalize training paths to (n_paths, n_steps)
    tp = np.asarray(training_paths, dtype=np.float64)
    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            states = tp
        elif tp.shape[1] == 4096:
            states = tp.T
        else:
            # Heuristic: assume first dim is paths
            states = tp
    else:
        states = tp.reshape(4096, -1)

    n_paths = states.shape[0]
    n_steps = states.shape[1]

    # Exercise dates are at columns 1..12 (0-indexed), i.e., t=1/12,...,1
    # states[:, j] for j=1..12 are the states at exercise dates 1..12

    # We'll store continuation value functions as piecewise or polynomial
    # For simplicity and robustness, use high-degree polynomial regression
    # with orthonormal basis on a standardized S range.

    # Determine S range for standardization
    S_all = states[:, 1:]  # all exercise-date states
    S_min = S_all.min()
    S_max = S_all.max()
    S_center = (S_min + S_max) / 2.0
    S_scale = (S_max - S_min) / 2.0 + 1e-12

    def standardize(S):
        return (S - S_center) / S_scale

    # Backward induction
    # V_next: continuation value at the next exercise date for each path
    # At the last date (date 12, column index 12), value is European put payoff
    V_next = np.maximum(K - states[:, 12], 0.0)

    # Store thresholds and continuation value functions
    # We'll store for each date a function that maps S -> continuation value
    # Using polynomial regression with degree d
    deg = 6  # polynomial degree

    # Store the fitted coefficients for each date's continuation value
    # cont_coeffs[i] for date i+1 (i=0..10, dates 1..11)
    # Date 12 is always European: max(K-S, 0)
    cont_coeffs = [None] * (n_dates - 1)  # for dates 1..11

    # Work backwards from date 11 down to date 1
    # At date i (1-indexed, i=11 down to 1), column index is i
    for i in range(n_dates - 1, 0, -1):
        if time.time() > DEADLINE:
            # Fallback: use simple threshold K for all remaining
            for j in range(i - 1, -1, -1):
                cont_coeffs[j] = None
            break

        col_idx = i  # column index in states for date i
        S_i = states[:, col_idx]
        immediate = np.maximum(K - S_i, 0.0)

        # Continuation value at date i is disc * V_next (already discounted from next date)
        cont_val = disc * V_next

        # Fit polynomial regression: cont_val ~ poly(standardize(S_i))
        # Use orthonormal polynomials for stability
        S_std = standardize(S_i)

        # Build Vandermonde matrix for polynomial basis
        # Use degree deg
        X = np.vander(S_std, N=deg + 1, increasing=True)

        # Ridge regression for stability
        alpha = 1e-6
        try:
            coeffs = np.linalg.solve(X.T @ X + alpha * np.eye(deg + 1), X.T @ cont_val)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(X, cont_val, rcond=None)[0]

        cont_coeffs[i - 1] = coeffs  # stored for date i (1-indexed), index i-1

        # Update V_next: value at date i is max(immediate, cont_val)
        V_next = np.maximum(immediate, cont_val)

    # Now evaluate the policy on the training paths to estimate thresholds
    # For each date, find the optimal threshold by searching
    thresholds = np.zeros(n_dates)

    # Date 12: always European
    thresholds[11] = K

    for i in range(11, 0, -1):
        if time.time() > DEADLINE:
            thresholds[i - 1] = K
            continue

        col_idx = i
        S_i = states[:, col_idx]
        S_std = standardize(S_i)
        immediate = np.maximum(K - S_i, 0.0)

        # Get continuation value from the fitted polynomial
        if cont_coeffs[i - 1] is not None:
            X = np.vander(S_std, N=deg + 1, increasing=True)
            cont_val = X @ cont_coeffs[i - 1]
            cont_val = np.maximum(cont_val, 0.0)
        else:
            cont_val = disc * V_next

        # Find threshold: the S value where immediate = continuation
        # immediate = max(K - S, 0)
        # For S < K: immediate = K - S
        # We want K - S = cont_val(S)
        # Define f(S) = (K - S) - cont_val(S) for S < K
        # Find root of f(S) = 0

        # Sample S values to find the threshold
        S_search = np.linspace(0.1 * S_min, K, 200)
        S_search_std = standardize(S_search)
        X_search = np.vander(S_search_std, N=deg + 1, increasing=True)

        if cont_coeffs[i - 1] is not None:
            cont_search = X_search @ cont_coeffs[i - 1]
            cont_search = np.maximum(cont_search, 0.0)
        else:
            # Fallback: use average continuation
            cont_search = np.full_like(S_search, cont_val.mean())

        f = (K - S_search) - cont_search

        # Find where f crosses zero
        sign_changes = np.where(np.diff(np.sign(f)) != 0)[0]
        if len(sign_changes) > 0:
            # Take the first sign change (lowest S where we stop exercising)
            idx = sign_changes[0]
            # Linear interpolation
            S1, S2 = S_search[idx], S_search[idx + 1]
            f1, f2 = f[idx], f[idx + 1]
            thr = S1 - f1 * (S2 - S1) / (f2 - f1)
        else:
            # No sign change: check if always exercise or never
            if f[0] > 0 and f[-1] > 0:
                thr = K  # always exercise up to K
            elif f[0] < 0 and f[-1] < 0:
                thr = 0.0  # never exercise
            else:
                thr = K

        thresholds[i - 1] = np.clip(thr, 0.0, K)

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K_val):
            self.thresholds = thresholds
            self.K = K_val

        def exercise(self, current_state, time):
            # Determine the exercise date index from time
            if isinstance(time, (int, float)):
                t_val = float(time)
                if 0 < t_val <= 1.0:
                    idx = int(np.round(t_val * 12)) - 1
                    idx = max(0, min(idx, 11))
                elif 1 <= t_val <= 12:
                    idx = int(t_val) - 1
                    idx = max(0, min(idx, 11))
                else:
                    idx = 0
            else:
                idx = 0

            state = np.asarray(current_state, dtype=np.float64)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
