"""Bermudan exercise-policy: high-resolution binomial tree backward induction with smooth refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # 12 monthly exercise dates

    # Use a high-resolution binomial tree for accurate continuation values
    n_steps = 1200
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    S0 = 100.0  # reference spot for tree construction

    # First pass: compute value functions at each exercise date using backward induction
    # We store the continuation value (before exercise decision) at each exercise date
    i = n_steps
    j = np.arange(i + 1)
    S_final = S0 * u ** (i - 2 * j)
    V = np.maximum(K - S_final, 0.0)

    # Store continuation values at exercise dates for refinement
    cont_values = {}
    cont_values[12] = (S_final.copy(), np.maximum(K - S_final, 0.0).copy())

    for i in range(n_steps - 1, -1, -1):
        V_cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])

        if i > 0 and i % 100 == 0:
            j_curr = np.arange(i + 1)
            S_curr = S0 * u ** (i - 2 * j_curr)
            V_exercise = np.maximum(K - S_curr, 0.0)
            V = np.maximum(V_cont, V_exercise)
            exercise_date = i // 100
            cont_values[exercise_date] = (S_curr.copy(), V_cont.copy())
        else:
            V = V_cont

        if time.time() > deadline:
            break

    # Now we have the tree-based policy. Let's refine it using a smooth approach.
    # Extract the critical boundaries from the tree for each exercise date.
    # For a put, the exercise region is S <= S_crit.
    # From the tree, at each exercise date, we can find the threshold.
    
    # Let's extract initial boundaries from the tree
    boundaries = np.zeros(n_exercise)
    for k in range(1, n_exercise + 1):
        if k in cont_values:
            S_nodes, V_cont_nodes = cont_values[k]
            # Find where exercise is optimal: K - S >= V_cont
            # Since S_nodes is in decreasing order, find the transition point
            exercise_payoff = np.maximum(K - S_nodes, 0.0)
            exercise_mask = exercise_payoff >= V_cont_nodes
            # Find the last index where exercise is optimal
            if np.any(exercise_mask):
                last_exercise_idx = np.where(exercise_mask)[0][-1]
                # The boundary is between S_nodes[last_exercise_idx] and S_nodes[last_exercise_idx+1]
                if last_exercise_idx < len(S_nodes) - 1:
                    # Interpolate to find the exact crossing
                    S1 = S_nodes[last_exercise_idx]
                    S2 = S_nodes[last_exercise_idx + 1]
                    V1 = V_cont_nodes[last_exercise_idx]
                    V2 = V_cont_nodes[last_exercise_idx + 1]
                    # Linear interpolation
                    if abs(V1 - V2) > 1e-10:
                        # K - S = V_cont
                        # K - S1 = V1, K - S2 = V2
                        # We want S where K - S = V_cont(S)
                        # Linear approx: V_cont(S) ≈ V1 + (V2-V1)*(S-S1)/(S2-S1)
                        # K - S = V1 + (V2-V1)*(S-S1)/(S2-S1)
                        # K - S - V1 = (V2-V1)*(S-S1)/(S2-S1)
                        # (K - V1) - S = (V2-V1)*(S-S1)/(S2-S1)
                        # Let's just use the midpoint or the S1 value
                        boundaries[k-1] = S1
                    else:
                        boundaries[k-1] = S1
                else:
                    boundaries[k-1] = S_nodes[last_exercise_idx]
            else:
                boundaries[k-1] = 0.0
        else:
            boundaries[k-1] = 0.0

    # Now refine the boundaries using a smooth optimization approach
    # We'll use a surrogate objective that smooths the non-differentiable payoff
    # The objective is to maximize the expected discounted payoff
    
    # Exercise dates
    exercise_dates = np.linspace(0, T, n_exercise + 1)[1:]
    
    # Precompute discount factors
    disc_factors = np.exp(-r * exercise_dates)
    
    n_paths = training_paths.shape[0]
    
    # Smooth refinement using log-sum-exp based surrogate
    # The key insight: instead of hard threshold, use a soft exercise probability
    # This allows gradient-based optimization
    
    def smooth_payoff(bnds, temperature=5.0):
        """Compute expected payoff with smooth exercise decision."""
        payoffs = np.zeros(n_paths)
        not_exercised = np.ones(n_paths, dtype=bool)
        
        for i in range(n_exercise):
            if not np.any(not_exercised):
                break
            S_i = training_paths[:, i]
            
            # Smooth exercise decision: probability of exercise
            # Use sigmoid-like function
            diff = bnds[i] - S_i
            # Soft decision: 1 if exercise, 0 if not, smoothed
            # Use tanh for smoothness
            exercise_prob = 0.5 * (1 + np.tanh(diff / temperature))
            
            # For paths that haven't been exercised yet
            mask = not_exercised
            if np.any(mask):
                # Expected payoff from exercising now (smoothed)
                exercise_now = (K - S_i[mask]) * exercise_prob[mask]
                # Discounted
                payoffs[mask] += exercise_now * disc_factors[i]
                # Update not_exercised: paths that "exercised" are removed
                # For smooth version, we reduce the weight
                # Actually, for a proper smooth approximation, we need to track
                # the probability of not having exercised yet
                # This is complex. Let's use a simpler approach:
                # Just use the hard decision for the mask, but smooth the boundary
                hard_exercise = S_i[mask] < bnds[i]
                payoffs[mask[hard_exercise]] = (K - S_i[mask[hard_exercise]]) * disc_factors[i]
                not_exercised[mask[hard_exercise]] = False
        
        if np.any(not_exercised):
            S_T = training_paths[:, -1]
            payoff_T = np.maximum(K - S_T, 0.0)
            payoffs[not_exercised] = payoff_T[not_exercised] * disc_factors[-1]
        
        return payoffs

    # Gradient-based refinement
    # Use central differences for gradient estimation
    lr = 0.5
    n_iterations = 30
    eps = 0.3
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        grads = np.zeros(n_exercise)
        
        for i in range(n_exercise):
            if time.time() > deadline:
                break
            
            bnds_plus = boundaries.copy()
            bnds_plus[i] += eps
            bnds_minus = boundaries.copy()
            bnds_minus[i] -= eps
            
            mean_plus = np.mean(smooth_payoff(bnds_plus))
            mean_minus = np.mean(smooth_payoff(bnds_minus))
            
            grads[i] = (mean_plus - mean_minus) / (2.0 * eps)
        
        boundaries += lr * grads
        boundaries = np.clip(boundaries, 1.0, K * 1.5)
        
        # Adaptive learning rate
        lr *= 0.95

    # Final high-precision polish: use the tree to verify and adjust boundaries
    # For each exercise date, use the tree to find the exact boundary
    # and blend with the refined boundary
    
    final_boundaries = np.zeros(n_exercise)
    for k in range(1, n_exercise + 1):
        if k in cont_values:
            S_nodes, V_cont_nodes = cont_values[k]
            # Find the exact boundary from the tree
            exercise_payoff = np.maximum(K - S_nodes, 0.0)
            exercise_mask = exercise_payoff >= V_cont_nodes
            if np.any(exercise_mask):
                last_exercise_idx = np.where(exercise_mask)[0][-1]
                if last_exercise_idx < len(S_nodes) - 1:
                    # Use the refined boundary if it's in the tree range
                    S_min_tree = S_nodes[-1]
                    S_max_tree = S_nodes[0]
                    if S_min_tree <= boundaries[k-1] <= S_max_tree:
                        final_boundaries[k-1] = boundaries[k-1]
                    else:
                        final_boundaries[k-1] = S_nodes[last_exercise_idx]
                else:
                    final_boundaries[k-1] = S_nodes[last_exercise_idx]
            else:
                final_boundaries[k-1] = 0.0
        else:
            final_boundaries[k-1] = boundaries[k-1]

    # Define the policy using the refined boundaries
    class BermudanPolicy:
        def __init__(self, boundaries, K, exercise_dates):
            self.boundaries = boundaries
            self.K = K
            self.exercise_dates = exercise_dates

        def exercise(self, current_state, time):
            # Determine which exercise date we're at
            if isinstance(time, (int, np.integer)):
                k = int(time)
            else:
                k = int(np.round(time * 12))
                k = max(1, min(12, k))
            
            if k < 1 or k > 12:
                return np.zeros(len(current_state), dtype=bool)
            
            boundary = self.boundaries[k-1]
            return current_state < boundary

    return BermudanPolicy(final_boundaries, K, exercise_dates)
# EVOLVE-BLOCK-END
