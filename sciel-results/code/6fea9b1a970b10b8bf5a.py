"""Bermudan put exercise policy via Monte Carlo Longstaff-Schwartz backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.linalg import solve


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Parse contract parameters
    S0 = float(contract.get('S0', 100.0))
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_dates = int(contract.get('n_dates', 12))

    n_train = training_paths.shape[0]
    # training_paths: shape (n_train, n_steps+1) or (n_train, n_dates) ?
    # Assume training_paths has prices at each exercise date.
    # We need to figure out the number of columns.
    n_cols = training_paths.shape[1]

    # Exercise dates are at t_1, t_2, ..., t_n (no time-0 exercise)
    # If n_cols == n_dates, then training_paths[:, j] is price at date j (1-indexed)
    # If n_cols == n_dates + 1, then training_paths[:, 0] is S0 and [:, j] is date j
    # Let's assume the columns correspond to the exercise dates in order.
    # The task says "12 monthly exercise dates (no time-0 exercise)".
    # So we have n_dates exercise dates.

    # Determine the time step
    dt = T / n_dates
    discount = np.exp(-r * dt)

    # Identify which columns are exercise dates
    # We'll assume training_paths has n_dates columns for the exercise dates
    # If it has n_dates+1, the first is time 0
    if n_cols == n_dates:
        # Columns 0..n_dates-1 correspond to dates 1..n_dates
        date_cols = list(range(n_dates))
    elif n_cols == n_dates + 1:
        # Column 0 is time 0, columns 1..n_dates are dates 1..n_dates
        date_cols = list(range(1, n_dates + 1))
    else:
        # Fallback: use last n_dates columns
        date_cols = list(range(n_cols - n_dates, n_cols))

    # Extract prices at each exercise date
    # prices_at_date[j] for j in 0..n_dates-1, corresponding to date j+1
    prices = []
    for j in range(n_dates):
        col = date_cols[j]
        prices.append(training_paths[:, col])
    prices = np.array(prices)  # shape (n_dates, n_train)

    # Payoff at each date: max(K - S, 0) for a put
    payoffs = np.maximum(K - prices, 0.0)  # shape (n_dates, n_train)

    # Backward induction
    # V[j] = continuation value at date j (before exercise decision)
    # We start from the last date: V[n_dates-1] = payoff[n_dates-1]
    # Then for j = n_dates-2 down to 0:
    #   V[j] = max(payoff[j], discount * E[V[j+1] | S_j])
    #   The expectation is approximated by regression on S_j

    # We'll store the regression coefficients for each date
    # deg: polynomial degree for regression on log(S)
    deg = 5

    n = n_train

    # V_next: value at next date (initially the last date's payoff)
    V_next = payoffs[-1].copy()  # shape (n_train,)

    # Store coefficients for each date (we need them for the exercise method)
    # coeffs[j] are the polynomial coefficients for date j
    # We also need to store which paths exercised at each date for the backward pass
    # But for the exercise method, we only need the regression coefficients.

    # We'll fit from last to first.
    # For date j (0-indexed), we need to fit V_next as a function of S_j.
    # We use Ridge Regression to stabilize the polynomial fit and reduce variance.
    # We also clip the fitted continuation values to be non-negative.

    coeffs = [None] * n_dates
    ridge_lambda = 1e-4  # Regularization parameter

    for j in range(n_dates - 1, -1, -1):
        if time.time() > deadline:
            break

        if j == n_dates - 1:
            # Last date: value is payoff, no regression needed
            coeffs[j] = None
            continue

        # Fit regression: V_next = f(S_j)
        S_j = prices[j]  # shape (n_train,)
        # Use log(S) for better conditioning
        logS = np.log(S_j + 1e-10)

        # Build design matrix for polynomial basis
        # Columns: 1, logS, (logS)^2, ..., (logS)^deg
        X = np.vander(logS, deg + 1, increasing=True)
        
        # Ridge Regression: (X^T X + lambda * I) c = X^T y
        # We don't penalize the intercept term (first column) usually, or we can.
        # Let's penalize all terms except the intercept for stability.
        # Actually, standard Ridge penalizes all. Let's stick to standard.
        # But to allow the mean to be fitted well, maybe don't penalize intercept?
        # Let's try standard Ridge first.
        
        try:
            XtX = X.T @ X
            Xty = X.T @ V_next
            
            # Add ridge penalty
            # Create a matrix that is 0 for the intercept (if we want to exclude it) or lambda for all.
            # Let's exclude the intercept from penalty to ensure the baseline level is fitted.
            I = np.eye(deg + 1)
            I[0, 0] = 0.0 # No penalty on intercept
            A = XtX + ridge_lambda * I
            b = Xty
            
            c = solve(A, b)
            coeffs[j] = c
        except Exception:
            # Fallback: use a simpler model (OLS)
            try:
                c = np.polynomial.polynomial.polyfit(logS, V_next, deg)
                coeffs[j] = c
            except Exception:
                coeffs[j] = None

        # Update V_next: at date j, the value is max(payoff_j, discount * fitted_continuation)
        if coeffs[j] is not None:
            fitted_cont = np.polynomial.polynomial.polyval(logS, coeffs[j])
            # Option values cannot be negative. Clipping reduces bias from negative predictions.
            fitted_cont = np.maximum(fitted_cont, 0.0)
        else:
            # Fallback: use mean of V_next as continuation
            fitted_cont = np.full(n_train, np.mean(V_next))

        V_j = np.maximum(payoffs[j], discount * fitted_cont)
        V_next = V_j

    # Now build the exercise policy object
    # The exercise method receives current_state (spot prices) and time (which date)
    # It should return a boolean vector: True if exercise

    class BermudanExercisePolicy:
        def __init__(self, coeffs, prices, payoffs, n_dates, K, discount, deg):
            self.coeffs = coeffs
            self.n_dates = n_dates
            self.K = K
            self.discount = discount
            self.deg = deg

        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices (shape (n_eval,))
            time_idx: which exercise date (0-indexed, 0..n_dates-1)
            Returns: boolean array, True if exercise
            """
            if time_idx >= self.n_dates:
                # Beyond last date, don't exercise
                return np.zeros(len(current_state), dtype=bool)

            S = np.asarray(current_state, dtype=float)
            payoff = np.maximum(self.K - S, 0.0)

            if time_idx == self.n_dates - 1:
                # Last date: exercise if payoff > 0
                return payoff > 0

            c = self.coeffs[time_idx]
            if c is None:
                # No regression: exercise if payoff > 0
                return payoff > 0

            logS = np.log(S + 1e-10)
            fitted_cont = np.polynomial.polynomial.polyval(logS, c)
            # Clip continuation value to be non-negative, consistent with training
            fitted_cont = np.maximum(fitted_cont, 0.0)
            cont_value = self.discount * fitted_cont
            # Exercise if immediate payoff > continuation value
            return payoff > cont_value

    return BermudanExercisePolicy(coeffs, prices, payoffs, n_dates, K, discount, deg)

# EVOLVE-BLOCK-END
