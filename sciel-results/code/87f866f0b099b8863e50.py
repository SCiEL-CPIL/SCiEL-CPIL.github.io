# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using a time-homogeneous optimal stopping boundary.
    The boundary is determined by solving for the critical price S_t* where the 
    intrinsic value equals the expected discounted continuation value, assuming 
    a stationary policy structure. This avoids the high variance and bias of 
    time-dependent regression on small samples.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)

    # Handle input shape
    n_paths, n_cols = training_paths.shape
    if n_cols == n_dates:
        exercise_data = training_paths
    elif n_cols == n_dates + 1:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]

    S_vals = [exercise_data[:, t] for t in range(n_dates)]
    
    dt = T / n_dates
    discount = np.exp(-r * dt)

    # We assume a time-homogeneous boundary S* such that we exercise if S_t <= S*.
    # The optimal S* satisfies: Intrinsic(S*) = Discount * E[Continuation(S_next) | S_t = S*]
    # Since the policy is "exercise if S <= S*", the continuation value at S* is the 
    # expected value of the payoff if we wait one step, given that we are at S*.
    # However, the "wait" decision leads to future decisions.
    # A robust approximation for the critical price in a time-homogeneous setting 
    # (or early dates where the boundary is flat) is to match the immediate intrinsic 
    # to the discounted expected intrinsic of the *next* step, assuming the next 
    # step also follows the same rule. This is a fixed-point approximation.
    
    # Let V(S) = max(K-S, D * E[V(S_next) | S])
    # At the boundary S*, K-S* = D * E[V(S_next) | S=S*]
    # If S* is deep enough, V(S_next) is often intrinsic. If shallow, it's continuation.
    # Given the small T and high K relative to mean spot (90-110), the option is likely 
    # in-the-money or near it.
    
    # Strategy: Find S* by minimizing the discrepancy between the policy value 
    # computed via backward induction on the training paths for a given S*, 
    # and the theoretical expectation. 
    # Actually, a simpler and more robust structural approach:
    # Use the training paths to estimate the conditional expectation E[S_{t+1} | S_t] 
    # is not needed if we use the analytical GBM moments, but we have paths.
    # Let's estimate the "effective" continuation value for a given S* using the paths.
    
    # We will search for the optimal time-homogeneous boundary S*.
    # The objective is to maximize the total discounted payoff on the training paths.
    
    # Precompute logs for efficiency
    log_S_vals = [np.log(S_vals[t]) for t in range(n_dates)]
    
    # Define a function to evaluate a specific boundary S* on the training paths
    def evaluate_policy(S_star):
        total_value = 0.0
        for t in range(n_dates - 1, -1, -1):
            S_t = S_vals[t]
            intrinsic = np.maximum(K - S_t, 0.0)
            
            if t == n_dates - 1:
                V_t = intrinsic
            else:
                # We need the continuation value. 
                # In a time-homogeneous boundary policy, if S_t > S*, we wait.
                # The value of waiting is the discounted value of the optimal policy at t+1.
                # Since we are evaluating a fixed S* for all t, we can compute V_t 
                # recursively from the end.
                # However, this evaluation inside the search loop is O(N * Paths).
                # N=12, Paths=4096. 12*4096 = ~50k ops per evaluation.
                # If we do 100 evaluations, that's 5M ops. Very fast.
                
                # We need V_{t+1} to compute V_t.
                # So we must compute the vector of values for all t first?
                # No, V_t depends on V_{t+1}.
                # Let's compute the value vectors for all t for a given S*.
                
                # This function will return the total mean discounted payoff.
                pass
        
        # We need to compute the full value vector to get the total.
        # Let's implement the backward pass for a given S*.
        V_next = None
        total_discounted_value = 0.0
        # We sum the values at t=0? No, the task says "Maximize mean discounted payoff".
        # Usually this means the value at t=0 (the price).
        # Or the sum of payoffs? "mean discounted payoff" usually implies the initial value V_0.
        # Let's assume the objective is V_0.
        
        # Backward pass
        V_curr = np.maximum(K - S_vals[-1], 0.0)
        for t in range(n_dates - 2, -1, -1):
            S_t = S_vals[t]
            intrinsic = np.maximum(K - S_t, 0.0)
            
            # Continuation value: discount * V_next
            # Note: V_next is the value at t+1 for the *same paths*.
            # This is a "crude" Monte Carlo backward induction which suffers from 
            # path-dependence issues (the value at t+1 is realized on the specific path,
            # not the conditional expectation). 
            # However, for finding the *boundary*, this is often sufficient if the 
            # boundary is stable.
            
            # A better way: Estimate E[V_{t+1} | S_t] using the training paths.
            # But that requires regression again.
            
            # Let's stick to the structural boundary idea:
            # The optimal boundary S* is the one where the option is indifferent.
            # For a Bermudan put, the boundary is typically decreasing over time (or constant).
            # Let's assume a constant boundary S*.
            
            # To evaluate S*, we need the "true" continuation value.
            # We can approximate E[V_{t+1} | S_t] by averaging V_{t+1} over all paths 
            # where S_t is close to the current S_t? No, that's local regression.
            
            # Alternative Structural Approach:
            # Use the analytical solution for the boundary of an American/Bermudan put.
            # For a European put, there is no early exercise.
            # For an American put, the boundary is S* < K.
            # The critical price S* satisfies:
            # K - S* = D * (A * S*^alpha + B * S*^beta) ... (Analytical form)
            # This is complex.
            
            # Let's use a simpler heuristic:
            # The optimal exercise region for a put is S < S*.
            # We can estimate S* by finding the price where the intrinsic value 
            # equals the discounted expected value of the *intrinsic* payoff of the 
            # next step, assuming the next step is also exercised if in the money.
            # This underestimates continuation value (ignores the optionality of waiting 
            # further), so it will lead to *early* exercise (conservative).
            
            # To improve, we can iterate:
            # 1. Start with S* = K.
            # 2. Compute V_t using backward induction with the rule "Exercise if S <= S*".
            # 3. The "optimal" S* is where the value is maximized.
            
            pass
            
        # Let's implement the evaluation of a specific S* properly.
        # We compute V_0 for that policy.
        V = np.maximum(K - S_vals[-1], 0.0)
        for t in range(n_dates - 2, -1, -1):
            S_t = S_vals[t]
            intrinsic = np.maximum(K - S_t, 0.0)
            
            # The continuation value is the discounted V from the next step.
            # In a Monte Carlo simulation, if we just discount V_{t+1} from the *same path*,
            # we are not computing the conditional expectation E[V_{t+1}|S_t].
            # We are computing a path-dependent value.
            # This is valid for *simulation* of the policy, but not for *estimating* the value function.
            # However, if we want to *find* the best policy, we can evaluate the total payoff
            # of that policy on the training paths.
            
            # Policy: Exercise if S_t <= S*.
            # If we exercise, we get intrinsic.
            # If we don't, we get the value of the option at t+1.
            # The value at t+1 is determined by the same rule.
            
            # So, V_t(path) = intrinsic(path) if S_t(path) <= S* else discount * V_{t+1}(path)
            
            # This recursive definition allows us to compute the total discounted payoff
            # for a given S* on the training paths.
            
            # Let's do this.
            if t == n_dates - 2:
                # V is currently V_{n-1}
                mask_exercise = S_t <= S_star
                V = np.where(mask_exercise, intrinsic, discount * V)
            else:
                mask_exercise = S_t <= S_star
                V = np.where(mask_exercise, intrinsic, discount * V)
                
        # The objective is the mean of V_0
        return np.mean(V)

    # Search for the best S*
    # S* should be in [0, K].
    # Let's sample candidate S* values.
    s_candidates = np.linspace(0.1, K, 200)
    best_s = K
    best_val = -np.inf
    
    # We can optimize this. The value function is concave in S*?
    # Let's just do a coarse grid search first, then refine.
    
    # Coarse search
    for s in s_candidates:
        if time.time() > deadline - 5:
            break
        val = evaluate_policy(s)
        if val > best_val:
            best_val = val
            best_s = s
            
    # Refine around best_s
    if best_val > -np.inf:
        lower = max(0.1, best_s - (s_candidates[1] - s_candidates[0]))
        upper = min(K, best_s + (s_candidates[1] - s_candidates[0]))
        s_refined = np.linspace(lower, upper, 50)
        for s in s_refined:
            if time.time() > deadline - 2:
                break
            val = evaluate_policy(s)
            if val > best_val:
                best_val = val
                best_s = s

    # Final Policy
    class ExercisePolicy:
        def __init__(self, K, S_star):
            self.K = K
            self.S_star = S_star

        def exercise(self, current_state, time):
            # Exercise if current state is below the boundary
            # Note: time is provided but we use a time-homogeneous boundary.
            # This is a structural assumption to reduce variance and overfitting.
            return current_state <= self.S_star

    return ExercisePolicy(K, best_s)

# EVOLVE-BLOCK-END
