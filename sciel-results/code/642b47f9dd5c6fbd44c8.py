# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using a parametric smooth boundary representation.
    The boundary is defined by a monotonic transformation (sigmoid of a polynomial in log-price)
    for each exercise date. Parameters are optimized via L-BFGS-B to maximize the
    simulated Bermudan put value on the training paths.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)

    # Handle input shape
    n_paths, n_cols = training_paths.shape
    if n_cols == n_dates:
        S = training_paths
    elif n_cols == n_dates + 1:
        S = training_paths[:, 1:n_dates+1]
    else:
        S = training_paths[:, :n_dates]

    dt = T / n_dates
    discount = np.exp(-r * dt)

    # Pre-compute log-prices for efficiency
    log_S = np.log(S)

    # Define the parametric boundary model
    # For each date t, the exercise threshold is a function of log_S.
    # We model the "exercise probability" or "decision" as:
    # exercise(t, S) = 1 if sigmoid(a_t * log_S + b_t) > 0.5
    # which simplifies to: a_t * log_S + b_t > 0  =>  log_S > -b_t/a_t (if a_t > 0)
    # However, a simple linear threshold might be too rigid.
    # Let's use a slightly more flexible but still smooth boundary:
    # The boundary value B_t is determined by a global polynomial in time?
    # No, let's keep it per-date but smooth in S.
    
    # Better approach:
    # The optimal boundary for a put is decreasing in S.
    # We can parameterize the boundary log-price threshold for each date t as:
    # log_B_t = c0 + c1 * t + c2 * t^2 ... ?
    # Or simply, we optimize the boundary points directly?
    # Direct optimization of 12 boundary points is a 12-dim problem.
    # Let's try that first: B_0, B_1, ..., B_11.
    # But we need to ensure they are ordered? Not strictly, but usually B_t is decreasing as t increases (closer to expiry, higher strike needed to exercise? No, for puts, as time to expiry decreases, the critical spot decreases. So B_t should decrease as t increases towards n_dates-1).
    # Actually, for a put, you exercise when S is low.
    # At t=0 (long time to expiry), you are less likely to exercise unless S is very low. So B_0 is low.
    # At t=11 (short time to expiry), you exercise if S < K. So B_11 is close to K.
    # So B_t should generally increase with t.
    
    # Let's parameterize the log-boundary for each date as a smooth function of time index t.
    # log_B_t = f(t)
    # We can fit f(t) with a low-order polynomial, e.g., quadratic.
    # log_B_t = p0 + p1 * (t / n_dates) + p2 * (t / n_dates)^2
    # This reduces the dimensionality to 3 parameters.
    # This enforces smoothness across time.
    
    # Objective function: Maximize the mean discounted payoff
    def compute_value(params):
        p0, p1, p2 = params
        # Time fractions
        t_frac = np.arange(n_dates) / n_dates
        # Log-boundaries for each date
        log_B = p0 + p1 * t_frac + p2 * t_frac**2
        B = np.exp(log_B)
        
        # Backward induction to compute value
        # V[t] is the value at date t given state S[t]
        # We need to compute the continuation value for each path.
        
        # Initialize value at last date
        V = np.zeros(n_paths)
        V[:] = np.maximum(K - S[:, -1], 0.0)
        
        # Iterate backwards
        for t in range(n_dates - 2, -1, -1):
            # Discounted continuation value
            # This is the expected value of V[t+1] given S[t] under the risk-neutral measure.
            # Since we are using training paths, we approximate the expectation by the average of the next step's values?
            # NO. This is the core issue. Simple averaging of next step values along the path is not the conditional expectation E[V_{t+1} | S_t].
            # The current seed program uses regression to estimate E[V_{t+1} | S_t].
            # If we just use the path's own next value, we are not doing regression, we are doing a "realized" path value, which is biased and high variance.
            
            # To make this work, we MUST estimate the conditional expectation E[V_{t+1} | S_t].
            # We can do this by regressing V[t+1] on S[t] using the training paths.
            # Since we are optimizing parameters, we need to do this regression inside the objective function.
            # This is expensive: 12 regressions per evaluation.
            # n_paths = 4096.
            # 4096 points, regression with few features. Fast.
            
            # Let's use a simple regression: V_{t+1} = a0 + a1 * S_t + a2 * S_t^2 ?
            # Or log-S?
            # Let's use log-S for stability.
            # Features: [1, log_S_t, (log_S_t)^2]
            
            X = np.column_stack([np.ones(n_paths), log_S[:, t], log_S[:, t]**2])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = np.array([np.mean(V), 0.0, 0.0])
            
            # Predicted continuation value
            cont_val = X @ coeffs
            disc_cont_val = cont_val * discount
            
            # Intrinsic value
            intrinsic_val = np.maximum(K - S[:, t], 0.0)
            
            # Exercise decision based on boundary
            # Exercise if S_t < B_t
            # B_t is the boundary spot price.
            # log_B_t = p0 + p1 * t_frac[t] + p2 * t_frac[t]**2
            # S_t < B_t  <=>  log_S_t < log_B_t
            
            t_frac_t = t / n_dates
            log_B_t = p0 + p1 * t_frac_t + p2 * t_frac_t**2
            
            exercise_mask = log_S[:, t] < log_B_t
            
            # Value at t
            V = np.where(exercise_mask, intrinsic_val, disc_cont_val)
            
        # The value of the policy is the mean of V at t=0?
        # No, the task says "Maximize mean discounted payoff".
        # Usually this means the value at time 0.
        # But wait, the contract says "12 monthly exercise dates (no time-0 exercise)".
        # So the first decision is at t=1?
        # "Spot 90/100/110" - these are likely the initial states?
        # "Candidate receives current states only during scoring"
        # The evaluation paths are independent.
        # The "mean discounted payoff" likely refers to the value of the option at the start.
        # However, if there is no time-0 exercise, the value at time 0 is the discounted expected value of the value at time 1 (or the first exercise date).
        # Let's assume the value we want to maximize is the mean of V[:, 0] if t=0 is the first date provided.
        # The input `training_paths` has `n_dates` columns.
        # If `n_dates` is 12, and "no time-0 exercise", does column 0 correspond to t=1 month?
        # The contract says "12 monthly exercise dates".
        # Usually, paths are generated for the exercise dates.
        # So S[:, 0] is the spot at the first exercise date.
        # The value of the policy is the value at that first date, discounted back to time 0?
        # Or is the value at the first date the objective?
        # "Maximize mean discounted payoff"
        # If the evaluator calculates the value at time 0, it would discount V[:, 0] by exp(-r * dt).
        # Let's maximize the mean of V[:, 0] * discount? Or just V[:, 0]?
        # Since the discount factor is constant, maximizing V[:, 0] is equivalent.
        # Let's return the mean of V[:, 0].
        
        return -np.mean(V) # Negate for minimization

    # Initial guess for parameters
    # B_t should be around K * exp(-sigma^2 * T/2 ...)?
    # For a put, boundary is below K.
    # Let's start with a constant boundary at K * 0.8
    log_K = np.log(K)
    p0_init = log_K - 0.2
    p1_init = 0.0
    p2_init = 0.0
    x0 = np.array([p0_init, p1_init, p2_init])
    
    # Optimization
    # Use L-BFGS-B
    # We need to be careful about the number of evaluations.
    # Each evaluation involves 12 lstsq calls.
    # 4096 x 3 matrix.
    # Should be fast.
    
    best_x = x0
    best_val = compute_value(x0)
    
    # Check time
    if time.time() > deadline - 20:
        # Not enough time to optimize, use initial guess
        pass
    else:
        try:
            # Maximize the number of iterations but respect time
            # We can't easily interrupt minimize, so we rely on the fact that it's fast.
            # If it's too slow, we might time out.
            # Let's limit iterations.
            res = minimize(compute_value, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            if res.success and res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    # Construct the policy object
    class ExercisePolicy:
        def __init__(self, K, params, n_dates, r, T):
            self.K = K
            self.params = params
            self.n_dates = n_dates
            self.r = r
            self.T = T
            self.dt = T / n_dates
            self.discount = np.exp(-self.r * self.dt)
            self.p0, self.p1, self.p2 = params

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            """
            if time < 0 or time >= self.n_dates:
                return (self.K - current_state) > 0

            t_frac = time / self.n_dates
            log_B_t = self.p0 + self.p1 * t_frac + self.p2 * t_frac**2
            log_S = np.log(current_state)
            
            # Exercise if log_S < log_B_t
            return log_S < log_B_t

    return ExercisePolicy(K, best_x, n_dates, r, T)

# EVOLVE-BLOCK-END
