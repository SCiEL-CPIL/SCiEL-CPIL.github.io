import numpy as np
import time
from sklearn.linear_model import Ridge
import warnings

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = np.asarray(critical_prices)
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        # current_state: array-like
        # time: float
        # Determine exercise index
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        # Validate index
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return np.asarray(current_state) <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
        
    M = paths.shape[0]
    
    # Time steps
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    # We work backwards from T to 1/12
    # V at T (index 11) is the payoff
    V = np.maximum(K - paths[:, 11], 0.0)
    
    critical_prices = np.zeros(12)
    critical_prices[11] = K # At maturity, exercise if ITM, so crit price is K (actually, for a put, exercise if S < K, so crit is K)
    # Wait, at maturity, if S < K, we exercise. If S >= K, we don't.
    # The condition is S <= crit. If crit = K, then S <= K means exercise.
    # So critical_prices[11] = K is correct for the threshold logic S <= crit.
    
    # Basis functions for regression
    # We use polynomials of S.
    # To avoid numerical issues, we might normalize S or use features like S, S^2, ...
    # Let's use a set of features: 1, S, S^2, S^3, S^4, S^5
    # We can also add interaction terms if we had other state variables, but here it's just S.
    
    def get_features(S):
        # S is a 1D array
        S = np.asarray(S).flatten()
        # Normalize S to [0, 1] or similar to improve conditioning?
        # S is around 100. Let's use S/100.
        x = S / 100.0
        return np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            x**5
        ])
    
    # We will estimate the continuation value at each step k-1 by regressing V_k on S_{k-1}
    # V_k is the value at step k (which includes the exercise decision at k)
    # We regress the *discounted* V_k on S_{k-1}.
    
    # Loop backwards from k=12 (index 11) to k=1 (index 0)
    # At step k (1-indexed), we have prices paths[:, k-1] (current) and paths[:, k] (next)
    # Wait, the paths are given as (M, 12) corresponding to t=1/12, ..., 12/12.
    # So paths[:, 11] is S_T.
    # V_T = payoff at T.
    # We want to find critical price at t=11/12 (index 10).
    # We regress V_T (discounted by dt=1/12) on S_{11/12} (paths[:, 10]).
    
    dt = 1.0 / 12.0
    disc_step = np.exp(-r * dt)
    
    # Initialize V_next as the payoff at T
    V_next = np.maximum(K - paths[:, 11], 0.0)
    
    # critical_prices[11] corresponds to t=1.0 (index 11 in 1-based, 11 in 0-based)
    # The exercise decision at t=1.0 is just the payoff.
    # The critical price at t=1.0 is K.
    
    for k_idx in range(10, -1, -1):
        # k_idx is 0-based index in paths array, corresponding to time (k_idx+1)/12
        # We are estimating the continuation value at time (k_idx+1)/12
        # by regressing the discounted value at time (k_idx+2)/12 on S at time (k_idx+1)/12
        
        S_curr = paths[:, k_idx]
        
        # Discount the next value
        V_next_disc = disc_step * V_next
        
        # Build features
        X = get_features(S_curr)
        
        # Fit Ridge regression
        # Use a small alpha for regularization
        try:
            model = Ridge(alpha=1.0)
            model.fit(X, V_next_disc)
            V_cont_est = model.predict(X)
        except Exception:
            # Fallback to simple mean if regression fails
            V_cont_est = np.full_like(S_curr, np.mean(V_next_disc))
        
        # Determine exercise decision
        # Exercise if Payoff >= V_cont_est
        payoff_curr = np.maximum(K - S_curr, 0.0)
        exercise_mask = payoff_curr >= V_cont_est
        
        # Update V for this step
        V_curr = np.where(exercise_mask, payoff_curr, V_cont_est)
        
        # Find critical price
        # We want the largest S where exercise is optimal.
        # Since the policy is S <= crit, we can find the max S in the exercise set.
        # However, the regression might be noisy.
        # A more robust way: find the S where payoff(S) = V_cont(S).
        # Since we only have discrete points, we can take the max S where exercise is True.
        # But if the boundary is steep, this might be inaccurate.
        # Let's assume the boundary is well-approximated by the max exercised S.
        
        if np.any(exercise_mask):
            crit_price = np.max(S_curr[exercise_mask])
        else:
            crit_price = 0.0
            
        critical_prices[k_idx] = crit_price
        
        # Update V_next for the next iteration (backwards)
        V_next = V_curr
        
        # Time check
        if time.time() > deadline - 10.0:
            break

    # Optional: Refine critical prices using the full training set payoff
    # The LSMC estimate might be slightly biased due to the regression.
    # We can do a quick local search around these critical prices.
    
    # Let's evaluate the mean payoff with the current critical prices
    def calc_mean_payoff(c_prices):
        W = paths <= c_prices[np.newaxis, :]
        has_exercise = np.any(W, axis=1)
        first_idx = np.argmax(W, axis=1)
        payoffs = np.zeros(M)
        if np.any(has_exercise):
            ex_idx = np.where(has_exercise)[0]
            ex_steps = first_idx[ex_idx]
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            payoffs[ex_idx] = disc_ex * (K - S_ex)
        return np.mean(payoffs)
    
    # Try a few small perturbations?
    # Given the time constraint, and that LSMC is usually good, we might skip this.
    # But a simple gradient-free step could help.
    # Let's try a few random perturbations and keep the best.
    
    best_val = calc_mean_payoff(critical_prices)
    best_prices = critical_prices.copy()
    
    # If we have time, do a few random restarts/refinements
    if time.time() < deadline - 20.0:
        np.random.seed(42)
        for _ in range(10):
            # Perturb critical prices slightly
            noise = np.random.normal(0, 1.0, 12)
            new_prices = critical_prices + noise
            new_prices = np.clip(new_prices, 0, K)
            
            # Ensure monotonicity? For a put, crit prices should generally increase with time?
            # No, for a put, S_crit(t) decreases as t increases?
            # At T, S_crit = K. As t -> 0, S_crit -> 0?
            # Actually, for a deep ITM put, you exercise early to capture the interest.
            # So S_crit(t) is low for small t?
            # Let's check: If S is very low, you exercise immediately.
            # If S is high, you wait.
            # The boundary S_c(t) is the price below which you exercise.
            # At t=T, S_c(T)=K.
            # As t decreases, S_c(t) decreases.
            # So critical_prices[0] (t=1/12) < critical_prices[1] ... < critical_prices[11] (t=12/12=K).
            # Let's enforce this monotonicity.
            
            # Sort and enforce monotonicity
            # The indices are 0..11 corresponding to t=1/12..12/12.
            # We want new_prices[0] <= new_prices[1] <= ... <= new_prices[11]
            
            # Enforce monotonicity
            for i in range(1, 12):
                if new_prices[i] < new_prices[i-1]:
                    new_prices[i] = new_prices[i-1]
            
            val = calc_mean_payoff(new_prices)
            if val > best_val:
                best_val = val
                best_prices = new_prices
        
        critical_prices = best_prices

    return ExercisePolicy(critical_prices)
