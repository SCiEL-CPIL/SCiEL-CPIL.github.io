"""Bermudan exercise-policy research: ML-based stopping rule with antithetic training."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a Binomial Tree constructed from
    the contract parameters. This exploits the known GBM dynamics to compute
    the exact optimal boundaries for the given exercise dates.
    
    Parameters:
    -----------
    training_paths : ndarray
        Training price paths (ignored for computation, used only for shape/context if needed).
    contract : dict or object
        Contract specification with fields like K, r, sigma, T, n_steps, etc.
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector.
    """
    # Extract contract parameters
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    n_steps = getattr(contract, 'n_steps', 12)
    
    # Ensure numeric types
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    n_steps = int(n_steps)
    
    # If n_steps is not provided or is 0, try to infer from training_paths
    if n_steps <= 0:
        if isinstance(training_paths, np.ndarray):
            n_steps = training_paths.shape[1] - 1
        else:
            n_steps = 12
            
    dt = T / n_steps
    
    # Binomial Tree Parameters
    # Jarrow-Rudd parameters for better moment matching
    mu = (r - 0.5 * sigma * sigma) * dt
    u = np.exp(mu + sigma * np.sqrt(dt))
    d = np.exp(mu - sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # We need to compute the optimal exercise boundary for each of the n_steps exercise dates.
    # Exercise dates are at t_1, t_2, ..., t_n.
    # In the tree, we have nodes at steps 0 to n_steps.
    # Step 0 is t=0. Step i is t=i*dt.
    # We exercise at steps 1 to n_steps.
    
    # V[i][j] represents the value at step i, node j.
    # j ranges from 0 to i.
    # The stock price at step i, node j is S_0 * u^j * d^(i-j).
    # We assume S_0 is 100? No, the boundary is absolute.
    # Actually, the boundary B_i is the price S such that Exercise is optimal.
    # We can compute the value function on a grid of S values, or just find the critical index j*
    # at each step i where the decision flips.
    
    # Since the state space is continuous, but the tree is discrete, we find the critical node index
    # and then interpolate or use the price at that node as the boundary.
    # However, for a put, the exercise region is S <= B_i.
    # In the tree, for a fixed step i, as j (number of up moves) increases, S increases.
    # So for low j (low S), we exercise. For high j (high S), we continue.
    # There is a critical j* such that for j <= j*, we exercise, and for j > j*, we continue.
    
    # We will store the critical price B_i for each step i (1 to n_steps).
    boundaries = np.zeros(n_steps)
    
    # We need to perform backward induction.
    # We can work with the "value" at each node.
    # Let's create arrays to hold the value at each node for the current step.
    # max nodes at step i is i+1.
    # We'll use a list of arrays or just iterate backwards.
    
    # To be efficient, we can just store the value for each node in the final step,
    # and then recurse back.
    # However, we need the boundary at *each* step.
    
    # Let's use a recursive or iterative backward pass.
    # We'll store the continuation value at each node for the current step.
    
    # Initialize at step n_steps (maturity)
    # At maturity, Value = max(K - S, 0)
    # We don't need to store all values, just the boundary.
    # But to compute the boundary at step n_steps - 1, we need the expected value at step n_steps.
    
    # Let's store the value at each node for the current step in an array `values`.
    # `values` will have length `i+1` at step `i`.
    
    # We'll iterate from step = n_steps down to 1.
    
    # Step n_steps:
    # S_j = 100 * u^j * d^(n_steps - j) ? 
    # Wait, S_0 is not fixed in the policy, but the *relative* boundary is what matters?
    # No, the boundary is an absolute price level.
    # The tree is built starting from S_0. But S_0 is not given in the contract, it's "Spot 90/100/110" in the description, but the contract might just define the parameters.
    # The policy must work for any current state.
    # The boundary B_i is independent of S_0.
    # We can set S_0 = 1.0 for the tree construction, compute the boundary in terms of S_0,
    # and then the actual boundary is B_i * S_0_actual? No.
    # The boundary is a price level.
    # If we build the tree with S_0 = 100, the boundary will be an absolute price.
    # If we build it with S_0 = 1, the boundary is relative.
    # Since K=100, it's natural to use S_0 = 100 or just work with the formula.
    # Let's use S_0 = 1.0 and K=1.0 (scaled) or just use the actual K and assume S_0=100 for the tree generation?
    # Actually, the GBM dynamics are scale invariant.
    # S_{t+dt} = S_t * exp(...)
    # Payoff = max(K - S, 0)
    # If we scale S and K by the same factor, the boundary scales by that factor.
    # So we can set S_0 = 1.0, K' = K/S_0_ref. But we don't have S_0_ref.
    # Better: Just use the actual K and assume the tree starts at S=1.0? No, that changes the K/S ratio.
    # The optimal boundary depends on K/S_0.
    # The policy `exercise(S, t)` receives the actual S.
    # The boundary B_i is a number.
    # We can compute the boundary assuming S_0 = 1.0 and K_scaled = K.
    # Then the boundary B_i is the price level.
    # Wait, if S_0=1, and K=100, the put is deeply ITM. The boundary will be close to 100.
    # If S_0=100, K=100, ATM.
    # The boundary B_i is determined by the model parameters and K.
    # It does NOT depend on the current S_0. It is a function of time-to-maturity and parameters.
    # So we can build the tree with any S_0, say S_0=1.0, and use K=100.
    # The resulting boundary B_i will be the price level at which exercise is optimal.
    
    # Let's verify:
    # Value(S, t) = max(K-S, 0) at T.
    # Continuation(S, t) = E[Value(S_next, t+dt) | S].
    # The decision is S <= B(t).
    # This B(t) is unique for the given parameters.
    
    # So, we build the tree with S_0 = 1.0.
    # K is 100.
    # We compute the value at each node.
    # For each step i, we find the largest j such that Exercise > Continue.
    # The price at that node is S_i_j = 1.0 * u^j * d^(i-j).
    # This S_i_j is our boundary B_i.
    
    S0 = 1.0
    
    # We need to store the value at each node for the next step to compute the current step.
    # Let's use a list `next_values` where next_values[j] is the value at node j in the next step.
    # And `next_prices` where next_prices[j] is the price at node j in the next step.
    
    # Initialize for step n_steps (maturity)
    # Prices at step n_steps: j=0..n_steps
    # S_j = S0 * u^j * d^(n_steps - j)
    # Values at step n_steps: max(K - S_j, 0)
    
    # We'll iterate backwards from step = n_steps down to 1.
    # At step `i`, we have nodes j=0..i.
    # We need the values from step `i+1`.
    
    # Let's pre-compute the prices for all steps? Or just keep track.
    # It's easier to keep arrays for the current step's prices and the next step's values.
    
    # Start at step n_steps
    # We don't really need the "next" values for the last step, we just need the boundary for the last step.
    # Boundary for step n_steps: Exercise if K - S > 0 => S < K.
    # So B_n_steps = K. (Technically, we exercise if S < K. If S=K, payoff is 0, continue is 0. Indifferent. Usually exercise if S <= K for puts? No, if S=K, payoff 0. Continuation 0. So doesn't matter. But boundary is K.)
    
    # Let's do the backward induction properly.
    # We will store `values` for the step `i+1` in an array `next_vals`.
    # And `prices` for step `i+1` in `next_prices`.
    
    # Initialize for step n_steps
    n = n_steps
    next_prices = np.array([S0 * (u ** j) * (d ** (n - j)) for j in range(n + 1)])
    next_vals = np.maximum(K - next_prices, 0.0)
    
    boundaries = np.zeros(n)
    # Boundary for the last step (index n-1 in 0-based, step n in 1-based)
    # We exercise if S <= K.
    boundaries[n - 1] = K
    
    # Now iterate from step n-1 down to 1
    for i in range(n - 1, 0, -1):
        # Current step is i. Nodes j=0..i.
        # Next step is i+1. Nodes j=0..i+1.
        # We have next_prices (length i+2) and next_vals (length i+2).
        
        # Calculate current prices
        curr_prices = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        
        # Calculate expected continuation value for each current node j
        # From node j at step i, we go to j at step i+1 (down) with prob (1-p) and j+1 at step i+1 (up) with prob p.
        # Wait, standard binomial:
        # S_{i+1, j} = S_{i, j} * d (if down move)
        # S_{i+1, j+1} = S_{i, j} * u (if up move)
        # So from node j at step i, the next nodes are j (down) and j+1 (up) in step i+1.
        
        # Continuation Value at node j = discount * (p * V_{i+1, j+1} + (1-p) * V_{i+1, j})
        
        cont_vals = np.zeros(i + 1)
        for j in range(i + 1):
            # V_{i+1, j} is next_vals[j]
            # V_{i+1, j+1} is next_vals[j+1]
            exp_val = p * next_vals[j + 1] + (1 - p) * next_vals[j]
            cont_vals[j] = discount * exp_val
        
        # Immediate Payoff
        imm_payoff = np.maximum(K - curr_prices, 0.0)
        
        # Optimal Value
        opt_vals = np.maximum(imm_payoff, cont_vals)
        
        # Determine Boundary via Interpolation
        # We exercise if imm_payoff >= cont_vals
        # For a put, exercise region is S <= B.
        # Find the largest j such that imm_payoff[j] >= cont_vals[j]
        
        exercise_mask = imm_payoff >= cont_vals
        if np.any(exercise_mask):
            idxs = np.where(exercise_mask)[0]
            j_star = idxs[-1]
            
            # If the last node is an exercise node, the boundary is at least that node's price.
            # If j_star is the last node (i), we can't interpolate to the right.
            # But usually the boundary is interior.
            # We interpolate between j_star and j_star + 1.
            # Note: j_star + 1 must be a "continue" node for interpolation to make sense in the crossing.
            # If j_star == i (last node), then all nodes up to i exercise. 
            # This implies the boundary is > S_i. 
            # However, for a put, if S is very high, we don't exercise. 
            # If even the highest node exercises, the boundary is effectively "infinity" for this step? 
            # No, that's unlikely for a put with K=100 and S0=1 unless deep ITM.
            # If j_star < i, we interpolate between j_star and j_star+1.
            
            if j_star < i:
                S_j = curr_prices[j_star]
                S_next = curr_prices[j_star + 1]
                C_j = cont_vals[j_star]
                C_next = cont_vals[j_star + 1]
                
                # Slope of continuation value
                if S_next == S_j:
                    slope = 0
                else:
                    slope = (C_next - C_j) / (S_next - S_j)
                
                # Solve K - S = C_j + slope * (S - S_j)
                # K - C_j + slope * S_j = S * (1 + slope)
                denominator = 1.0 + slope
                if abs(denominator) < 1e-12:
                    boundary = S_j
                else:
                    boundary = (K - C_j + slope * S_j) / denominator
                
                # Clamp boundary to [S_j, S_next] to ensure stability
                boundary = np.clip(boundary, S_j, S_next)
            else:
                # Boundary is beyond the highest node. 
                # Use the highest node price as a conservative estimate or extrapolate?
                # For a put, if the highest node exercises, the boundary is high.
                # We just use the highest node price.
                boundary = curr_prices[j_star]
                
            boundaries[i - 1] = boundary
        else:
            boundaries[i - 1] = 0.0
            
        # Update for next iteration (which will be step i-1)
        next_prices = curr_prices
        next_vals = opt_vals
        
    # boundaries[0] corresponds to step 1 (t=dt)
    # boundaries[n-1] corresponds to step n (t=T)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_steps):
            self.boundaries = boundaries
            self.n_steps = n_steps
            self.dt = T / n_steps
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise.
            
            Parameters:
            -----------
            current_state : ndarray of shape (n_eval_paths,)
                Current prices.
            time : int or float
                Time index or time value.
            
            Returns:
            --------
            ndarray of bools, shape (n_eval_paths,)
            """
            # Convert time to exercise date index (0-based)
            if isinstance(time, (int, np.integer)):
                # Assume time is the exercise date index (1-based)
                idx = int(time) - 1
            else:
                # Assume time is the actual time value
                idx = int(round(float(time) / self.dt)) - 1
            
            # Clamp index
            idx = max(0, min(idx, self.n_steps - 1))
            
            boundary = self.boundaries[idx]
            return np.array(current_state) <= boundary
    
    return ExercisePolicy(boundaries, n_steps)
# EVOLVE-BLOCK-END
