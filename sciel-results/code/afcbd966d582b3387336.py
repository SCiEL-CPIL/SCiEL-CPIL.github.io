"""Bermudan exercise-policy: Longstaff-Schwartz Monte Carlo with polynomial regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)

    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Convert to numpy array
    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = training_paths.shape[0]

    # For Longstaff-Schwartz, we need the paths at each exercise date
    # training_paths shape: (n_paths, n_steps + 1) typically, or (n_paths, n_steps)
    # Assume it's (n_paths, n_steps) where each column is a time step
    # If it's (n_paths, n_steps+1), we skip time 0
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(-1, 1)
    
    if training_paths.shape[1] > n_steps:
        # Use the last n_steps columns (exercise dates)
        paths = training_paths[:, :n_steps]
    else:
        paths = training_paths

    # Ensure we have enough paths
    if n_paths < 100:
        # If too few paths, use a simple rule
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time):
                return np.asarray(current_state) < self.K
        return SimplePolicy(K)

    # Number of polynomial features to use
    # Use features: 1, S, S^2, log(S), S*log(S)
    # Actually, let's use a more standard set: 1, S, S^2, S^3, log(S)
    # Or just use S and S^2 for stability
    # Let's use: 1, S, S^2, S^3, log(S)
    # But log(S) can be problematic if S is very small. Let's clip S.
    S_min = 1e-4
    paths_clipped = np.clip(paths, S_min, None)

    # Precompute features for each time step
    # features[t] has shape (n_paths, n_features)
    n_features = 5
    features = np.zeros((n_steps, n_paths, n_features))
    for t in range(n_steps):
        S = paths_clipped[:, t]
        features[t, :, 0] = 1.0
        features[t, :, 1] = S
        features[t, :, 2] = S ** 2
        features[t, :, 3] = S ** 3
        features[t, :, 4] = np.log(S)

    # Backward induction
    # V[t] is the estimated value at time t for each path
    # We start from the last time step (t = n_steps - 1, which is index n_steps-1)
    # At the last exercise date, the value is the payoff
    V = np.maximum(K - paths_clipped[:, n_steps - 1], 0.0)

    # Store exercise decisions
    exercise_decisions = [None] * n_steps

    # At the last step, exercise if payoff > 0 (which is always true for a put if S < K)
    # Actually, at the last step, you always exercise if S < K
    exercise_decisions[n_steps - 1] = paths_clipped[:, n_steps - 1] < K

    # For t from n_steps - 2 down to 0:
    for t in range(n_steps - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break

        # Regress V on features[t]
        # V is the value at time t+1, which we want to discount to get continuation value at time t
        # Actually, the continuation value at time t is discount * E[V[t+1] | S[t]]
        # So we regress V on features[t] to get E[V[t+1] | S[t]]
        
        # Use least squares
        X = features[t]  # (n_paths, n_features)
        y = V  # (n_paths,)
        
        try:
            # Solve least squares
            theta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except:
            # Fallback: simple linear regression on S
            S = paths_clipped[:, t]
            A = np.column_stack([np.ones_like(S), S])
            try:
                theta_full, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                # Pad theta to match n_features
                theta = np.zeros(n_features)
                theta[0] = theta_full[0]
                theta[1] = theta_full[1]
            except:
                # Just use the mean
                theta = np.zeros(n_features)
                theta[0] = np.mean(y)

        # Predicted continuation value (before discount)
        cont_pred = X @ theta  # (n_paths,)
        
        # Discounted continuation value
        cont_val = discount * cont_pred

        # Exercise value
        exercise_val = np.maximum(K - paths_clipped[:, t], 0.0)

        # Decision: exercise if exercise_val >= cont_val
        exercise_decision = exercise_val >= cont_val
        exercise_decisions[t] = exercise_decision

        # Update V: the value at time t is max(exercise, continuation)
        V = np.maximum(exercise_val, cont_val)

    # Now we have exercise_decisions for each time step
    # We need to create a policy that can be applied to new paths

    class ExercisePolicy:
        def __init__(self, n_steps, K, exercise_decisions, paths_clipped, features, n_features):
            self.n_steps = n_steps
            self.K = K
            self.exercise_decisions = exercise_decisions
            self.paths_clipped = paths_clipped
            self.features = features
            self.n_features = n_features

            # We need to store the regression coefficients for each time step
            # But we didn't store them above. Let's recompute or store them.
            # Actually, we can store the theta vectors during the backward induction.
            # Let's modify the approach: store thetas

        def exercise(self, current_state, time):
            current_state = np.asarray(current_state, dtype=np.float64)
            if time < 0 or time >= self.n_steps:
                return np.zeros(len(current_state), dtype=bool)

            # Use the stored decision for this time step
            # But the decision was computed on training paths.
            # We need to apply the same rule to new paths.
            # The rule is: exercise if S < threshold, where threshold is determined by the regression.
            # Actually, the decision is: exercise if max(K-S, 0) >= discount * (X @ theta)
            # For a new path, we compute X @ theta and compare.

            # But we didn't store theta. Let's fix this.
            pass

    # Let's redo this properly, storing the thetas
    thetas = [None] * n_steps

    V = np.maximum(K - paths_clipped[:, n_steps - 1], 0.0)
    exercise_decisions = [None] * n_steps
    exercise_decisions[n_steps - 1] = paths_clipped[:, n_steps - 1] < K
    thetas[n_steps - 1] = None  # Not needed for last step

    for t in range(n_steps - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break

        X = features[t]
        y = V
        
        try:
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except:
            S = paths_clipped[:, t]
            A = np.column_stack([np.ones_like(S), S])
            try:
                theta_full, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                theta = np.zeros(n_features)
                theta[0] = theta_full[0]
                theta[1] = theta_full[1]
            except:
                theta = np.zeros(n_features)
                theta[0] = np.mean(y)

        thetas[t] = theta
        cont_pred = X @ theta
        cont_val = discount * cont_pred
        exercise_val = np.maximum(K - paths_clipped[:, t], 0.0)
        exercise_decision = exercise_val >= cont_val
        exercise_decisions[t] = exercise_decision
        V = np.maximum(exercise_val, cont_val)

    class ExercisePolicy:
        def __init__(self, n_steps, K, thetas, n_features, discount):
            self.n_steps = n_steps
            self.K = K
            self.thetas = thetas
            self.n_features = n_features
            self.discount = discount

        def exercise(self, current_state, time):
            current_state = np.asarray(current_state, dtype=np.float64)
            n_current = len(current_state)
            if time < 0 or time >= self.n_steps:
                return np.zeros(n_current, dtype=bool)

            if time == self.n_steps - 1:
                return current_state < self.K

            theta = self.thetas[time]
            if theta is None:
                return current_state < self.K

            S = np.clip(current_state, 1e-4, None)
            # Compute features
            X = np.zeros((n_current, self.n_features))
            X[:, 0] = 1.0
            X[:, 1] = S
            X[:, 2] = S ** 2
            X[:, 3] = S ** 3
            X[:, 4] = np.log(S)

            cont_pred = X @ theta
            cont_val = self.discount * cont_pred
            exercise_val = np.maximum(self.K - current_state, 0.0)

            return exercise_val >= cont_val

    policy = ExercisePolicy(n_steps, K, thetas, n_features, discount)
    return policy
# EVOLVE-BLOCK-END
