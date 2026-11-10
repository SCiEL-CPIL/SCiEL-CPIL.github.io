"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using a high-resolution 
    binomial tree model.
    
    Strategy:
    1. Build a high-resolution binomial tree that accurately captures the 
       risk-neutral dynamics of the underlying asset.
    2. Perform backward induction on the tree to compute the exact optimal 
       exercise boundaries for a Bermudan put under the given parameters.
    3. Use these theoretical boundaries as the exercise policy.
    
    This approach leverages the fact that the binomial tree model provides 
    a near-exact solution for the optimal exercise policy under the GBM 
    assumption, which should generalize well to the evaluation paths.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin within 120s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Monthly exercise dates
    dt = T / n_exercises
    
    # Build a high-resolution binomial tree
    # Use a large number of steps for accuracy
    n_steps = 100000  # High resolution for accuracy
    
    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1.0 - p
    
    # We only need the boundaries at the 12 exercise dates
    # These correspond to specific steps in the fine grid
    # Exercise date j (0-indexed) is at time (j+1)*dt
    # In the fine grid, this is step (j+1)*n_steps/n_exercises
    
    exercise_steps = np.arange(1, n_exercises + 1) * (n_steps // n_exercises)
    
    # We will store the value function at each exercise date
    # V_j[i] = value at node i at step exercise_steps[j]
    # Node i corresponds to S = S_0 * u^(i) * d^(step - i)
    
    # We need to track the range of nodes that are ITM
    # For a put, ITM if S < K
    
    # Start from the last exercise date
    # At the last date, value = max(K - S, 0)
    
    # We need to be careful with memory. Let's store only the values at exercise dates.
    # But we need to propagate backwards.
    
    # Let's use a sparse representation or just arrays of reasonable size.
    # At step t, the number of nodes is t+1.
    # At step n_steps, that's 100001 nodes. This is fine.
    
    # Initialize value at the final step (which is not an exercise date, but T)
    # Actually, the last exercise date is at T.
    # So step n_steps corresponds to T.
    
    # Let's compute the stock prices at the final step
    # S_final[i] = S_0 * u^(i) * d^(n_steps - i)
    # S_0 is not given, but the tree is homogeneous.
    # The value function V(S) is what we want.
    # We can normalize by S_0 or just work with the ratio S/S_0.
    # Actually, the boundary B_j is an absolute price level.
    # So we need to assume a starting price S_0.
    # The contract doesn't specify S_0 for the tree, but the evaluation spots are 90/100/110.
    # The binomial tree is a model of the distribution, independent of the initial spot for the *policy* 
    # if we consider the policy as a function of current state S.
    # However, the tree nodes are defined relative to S_0.
    # Let's pick S_0 = 100 (K) for simplicity, or just work in log-space.
    
    # Let's work in log-space to make it cleaner.
    # ln(S) = ln(S_0) + (r - 0.5*sigma^2)*t + sigma*sqrt(t)*Z
    # In the binomial tree:
    # ln(S_t) = ln(S_0) + (m - 0.5*sigma^2)*t + sigma*sqrt(t)*X
    # where X is the number of up steps minus down steps? No.
    
    # Standard CRR tree:
    # S_t = S_0 * u^i * d^(t-i) = S_0 * exp(i * ln(u) + (t-i) * ln(d))
    # ln(S_t) = ln(S_0) + i * ln(u) + (t-i) * ln(d)
    # ln(u) = sigma*sqrt(dt), ln(d) = -sigma*sqrt(dt)
    # ln(S_t) = ln(S_0) + (2i - t) * sigma * sqrt(dt)
    
    # Let's define the nodes by their index i (number of up steps).
    # At step t, i ranges from 0 to t.
    # S(t, i) = S_0 * exp((2i - t) * sigma * sqrt(dt))
    
    # We want to find the boundary B_j such that we exercise if S <= B_j.
    # This corresponds to a maximum index i_max such that S(t, i) <= B_j.
    
    # Let's pick S_0 = 100.0.
    S_0 = 100.0
    
    # Precompute the log-step size
    log_step = sigma * np.sqrt(dt)
    
    # We will compute the value function backwards.
    # V[t, i] is the value at step t, node i.
    # We only care about values at exercise steps.
    # But we need to propagate through all steps.
    
    # To save memory and time, we can just store the current row and the next row.
    # But we need to stop and record the boundary at exercise dates.
    
    # Let's iterate backwards from t = n_steps to t = 0.
    # V_next is the value at step t+1.
    # V_curr is the value at step t.
    
    # At t = n_steps (T), V = max(K - S, 0)
    # S(n_steps, i) = S_0 * exp((2i - n_steps) * log_step)
    
    # We need to handle the fact that the grid might not align perfectly with the exercise dates.
    # n_steps = 100000, n_exercises = 12.
    # 100000 / 12 = 8333.33...
    # So the exercise dates don't align with integer steps.
    # We should choose n_steps to be a multiple of n_exercises.
    # Let n_steps = 12 * 8333 = 99996. Or 12 * 8334 = 100008.
    # Let's use n_steps = 99996.
    
    n_steps = 99996
    dt_fine = T / n_steps
    
    # Recalculate tree params for fine grid
    u_fine = np.exp(sigma * np.sqrt(dt_fine))
    d_fine = 1.0 / u_fine
    p_fine = (np.exp(r * dt_fine) - d_fine) / (u_fine - d_fine)
    q_fine = 1.0 - p_fine
    disc_fine = np.exp(-r * dt_fine)
    
    log_step_fine = sigma * np.sqrt(dt_fine)
    
    # Exercise steps in the fine grid
    # Exercise date j (0-indexed, 1..12) is at time (j+1)*T/12
    # Step index = (j+1)*T/12 / dt_fine = (j+1)*T/12 / (T/99996) = (j+1)*99996/12
    steps_per_exercise = n_steps // n_exercises
    exercise_steps_fine = [j * steps_per_exercise for j in range(1, n_exercises + 1)]
    
    # Initialize V at t = n_steps
    # i ranges from 0 to n_steps
    # This is a huge array (100k).
    # Let's see if we can optimize.
    # We only need to know the boundary, i.e., the max i such that we exercise.
    # For a put, we exercise if S <= B.
    # S is increasing with i.
    # So we exercise for i <= i_max.
    # The value function V(t, i) is continuous and monotonic.
    
    # Let's just compute the full row. 100k floats is 800KB. That's fine.
    
    V_next = np.zeros(n_steps + 1)
    
    # t = n_steps
    # S_i = S_0 * exp((2i - n_steps) * log_step_fine)
    i_vals = np.arange(n_steps + 1)
    S_vals = S_0 * np.exp((2 * i_vals - n_steps) * log_step_fine)
    V_next = np.maximum(K - S_vals, 0.0)
    
    boundaries = np.zeros(n_exercises)
    boundaries[-1] = K  # At maturity, exercise if ITM
    
    # We need to record the boundary at each exercise step.
    # Let's store them in a dict or list.
    # We are going backwards, so we will hit exercise steps in reverse order.
    
    exercise_boundaries = {}
    
    # Check if the last step is an exercise step
    if n_steps in exercise_steps_fine:
        # Find max i where V_next[i] > 0 (i.e., S < K)
        # Actually, at maturity, we exercise if S < K.
        # S_i is increasing.
        # Find the largest i such that S_i < K.
        mask = S_vals < K
        if np.any(mask):
            i_max = np.where(mask)[0][-1]
            boundaries[-1] = S_vals[i_max]
        else:
            boundaries[-1] = 0.0
        exercise_boundaries[n_steps] = boundaries[-1]
        
    # Backward induction
    # We need to be careful with the discounting.
    # The discount factor for one fine step is disc_fine.
    
    # We will iterate t from n_steps - 1 down to 0.
    # This is 100k iterations.
    # Each iteration involves:
    # 1. Computing S_vals for row t.
    # 2. Computing continuation value: disc_fine * (p_fine * V_next[1:] + q_fine * V_next[:-1])
    #    Note: V_curr[i] = disc * (p * V_next[i+1] + q * V_next[i])
    #    So we need to align the arrays.
    #    V_next has length t+2 (since step t+1 has t+2 nodes).
    #    V_curr has length t+1.
    #    V_curr[i] corresponds to V_next[i] and V_next[i+1].
    # 3. Intrinsic value: max(K - S_vals, 0)
    # 4. V_curr = max(Intrinsic, Continuation)
    # 5. Check if t is an exercise step. If so, find boundary.
    
    # Time check
    # 100k iterations in Python might be slow if not vectorized.
    # But each iteration is O(t). Total work is O(n_steps^2) = 10^10. Too slow.
    
    # We need a faster approach.
    # We can use the fact that we only need the boundary.
    # Or we can reduce the number of steps.
    # The cell finding said: "Increasing grid resolution from 20k to 100k... improves score".
    # But 100k steps with O(N^2) complexity is too slow for 120s in Python.
    # 100k steps * avg 50k ops = 5e9 ops. Even in C, that's seconds. In Python, it's minutes.
    
    # We must use a smaller grid or a different method.
    # Let's try n_steps = 2000.
    # 2000 steps * avg 1000 ops = 2e6 ops. Very fast.
    # Is 2000 steps accurate enough?
    # dt = 1/2000 = 0.0005.
    # sigma*sqrt(dt) = 0.2 * 0.022 = 0.0044.
    # This is a fine grid.
    
    # Let's restart with n_steps = 2000.
    
    n_steps = 2000
    dt_fine = T / n_steps
    u_fine = np.exp(sigma * np.sqrt(dt_fine))
    d_fine = 1.0 / u_fine
    p_fine = (np.exp(r * dt_fine) - d_fine) / (u_fine - d_fine)
    q_fine = 1.0 - p_fine
    disc_fine = np.exp(-r * dt_fine)
    log_step_fine = sigma * np.sqrt(dt_fine)
    
    steps_per_exercise = n_steps // n_exercises
    exercise_steps_fine = [j * steps_per_exercise for j in range(1, n_exercises + 1)]
    
    # Initialize V at t = n_steps
    i_vals = np.arange(n_steps + 1)
    S_vals = S_0 * np.exp((2 * i_vals - n_steps) * log_step_fine)
    V_next = np.maximum(K - S_vals, 0.0)
    
    boundaries = np.zeros(n_exercises)
    # At maturity (step n_steps), boundary is K
    # Find max i where S < K
    mask = S_vals < K
    if np.any(mask):
        i_max = np.where(mask)[0][-1]
        boundaries[-1] = S_vals[i_max]
    else:
        boundaries[-1] = 0.0
        
    exercise_boundaries = {n_steps: boundaries[-1]}
    
    # Backward induction
    # We need to map exercise steps to the correct boundary index.
    # exercise_steps_fine[j] is the step for exercise date j (1-indexed in list, 0-indexed in boundaries? No)
    # exercise_steps_fine = [1*steps, 2*steps, ..., 12*steps]
    # boundaries[0] is for date 1 (step 1*steps)
    # boundaries[11] is for date 12 (step 12*steps = n_steps)
    
    # Let's create a map from step to boundary index
    step_to_bnd_idx = {}
    for idx, step in enumerate(exercise_steps_fine):
        # idx 0..11 corresponds to boundaries[0..11]
        step_to_bnd_idx[step] = idx
        
    # We already set boundaries[-1] (idx 11) for step n_steps.
    
    for t in range(n_steps - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Compute S_vals for row t
        # i ranges from 0 to t
        i_curr = np.arange(t + 1)
        S_curr = S_0 * np.exp((2 * i_curr - t) * log_step_fine)
        
        # Compute continuation value
        # V_curr[i] = disc_fine * (p_fine * V_next[i+1] + q_fine * V_next[i])
        # V_next has length t+2
        cont = disc_fine * (p_fine * V_next[1:] + q_fine * V_next[:-1])
        
        # Intrinsic value
        intr = np.maximum(K - S_curr, 0.0)
        
        # Optimal value
        V_curr = np.maximum(intr, cont)
        
        # Check if t is an exercise step
        if t in step_to_bnd_idx:
            bnd_idx = step_to_bnd_idx[t]
            # Find max i where we exercise (V_curr == intr, i.e., intr >= cont)
            # Actually, we exercise if intr > cont.
            # Or if intr >= cont.
            # Let's find the largest i such that intr[i] >= cont[i]
            ex_mask = intr >= cont
            if np.any(ex_mask):
                i_max = np.where(ex_mask)[0][-1]
                boundaries[bnd_idx] = S_curr[i_max]
            else:
                boundaries[bnd_idx] = 0.0
                
        V_next = V_curr
        
    # Ensure monotonicity
    # For puts, boundaries should be non-decreasing as time approaches maturity?
    # Actually, for puts, the exercise boundary typically *decreases* as time to maturity decreases?
    # No, for puts, you exercise when S is low.
    # As time to maturity decreases, the option value decreases (less time for S to recover? No, for puts, less time means less chance for S to go up, so it's more valuable to hold? No.)
    # Let's think.
    # American put: Exercise boundary is lower when there is more time to maturity?
    # Actually, for puts, the boundary is *lower* when there is *more* time to maturity?
    # No, usually for puts, the boundary is *higher* when there is *less* time to maturity.
    # Let's check.
    # At maturity, B = K.
    # At t=0, B is lower than K.
    # So B(t) is increasing as t -> T.
    # So boundaries[0] (t=dt) < boundaries[1] (t=2dt) < ... < boundaries[11] (t=T).
    
    for j in range(1, n_exercises):
        if boundaries[j] < boundaries[j-1]:
            boundaries[j] = boundaries[j-1]
            
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices for all paths at this time step.
            time: the time step index (0-based, 0 to 11 for 12 dates)
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            B = self.boundaries[time]
            return current_state <= B
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END
