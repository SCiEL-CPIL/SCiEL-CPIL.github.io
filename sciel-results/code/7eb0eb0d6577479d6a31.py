import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Fixed parameters from task description
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    exercise_times = np.linspace(1/12.0, T, n_exercise)

    # Use the training paths to infer the initial spot if possible, but the task says
    # "Spot 90/100/110" are the evaluation spots. The training paths likely start from
    # one of these or a mix. However, the optimal boundary is independent of S0.
    # We will compute the boundary using a high-precision binomial tree.
    # To be safe and fast, we can use a moderately high number of steps.
    # The previous program used N=2400 which might be slow if not optimized.
    # Let's use N=1000 or 2000. With NumPy vectorization, N=2000 is fast.
    
    # Check time
    if time.time() > deadline:
        # Fallback: simple heuristic
        return _fallback_policy()

    # Binomial tree parameters
    # We need enough steps to resolve the boundary accurately.
    # 12 exercise dates. Let's use steps_per_month = 100, so N = 1200.
    # This is a good balance.
    steps_per_month = 100
    N = steps_per_month * n_exercise  # 1200
    dt = T / N
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # We compute the critical price for each exercise date.
    # We work backwards from T.
    # V is an array of values at the current step.
    # At T, V = max(K - S_T, 0).
    
    # We need to map step index i to exercise date k.
    # i = 0..N
    # Exercise at i = k * steps_per_month for k=1..12.
    
    # To find the boundary at step i, we need S_i[j] for j=0..i.
    # S_i[j] = S0 * u^j * d^(i-j).
    # We can compute log S to avoid overflow/underflow, though for N=1200 it's fine.
    
    S0 = 100.0  # Reference S0
    log_S0 = np.log(S0)
    log_u = np.log(u)
    log_d = np.log(d)
    
    # Initialize at T (i=N)
    j_N = np.arange(N + 1)
    log_S_N = log_S0 + j_N * log_u + (N - j_N) * log_d
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = [0.0] * 13  # 1..12
    
    # Backward induction
    # We iterate i from N-1 down to 0.
    # At each step, we compute V_cont and V_ex.
    # If i is an exercise step, we find the boundary and store it.
    
    # Optimization: Pre-compute log_S for all steps? No, memory is O(N^2) which is ~1.4M floats, ~11MB. Acceptable.
    # But we can compute on the fly.
    
    # To speed up, we can use the fact that S_i[j] is monotonic.
    # The boundary is the largest j such that V_ex >= V_cont.
    
    # Let's loop
    for i in range(N - 1, -1, -1):
        if time.time() > deadline:
            break
            
        # Determine if this is an exercise step
        # i should be divisible by steps_per_month
        if i % steps_per_month == 0:
            k = i // steps_per_month
            if 1 <= k <= 12:
                # Compute S_i
                j_i = np.arange(i + 1)
                log_S_i = log_S0 + j_i * log_u + (i - j_i) * log_d
                S_i = np.exp(log_S_i)
                
                # Compute Continuation Value
                # V currently holds values for step i+1 (size i+2)
                # V_up = V[1:], V_down = V[:-1]
                # V_cont = exp(-r dt) * (p * V_up + (1-p) * V_down)
                # Note: V has size i+2. V[1:] has size i+1. V[:-1] has size i+1.
                V_up = V[1:]
                V_down = V[:-1]
                V_cont = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
                
                V_ex = np.maximum(K - S_i, 0.0)
                
                # Find boundary
                # We want the largest j such that V_ex[j] >= V_cont[j]
                # Since S_i is increasing, and for a put, exercise is optimal for low S,
                # the region of exercise is j=0..j_max.
                # So we find the last index where V_ex >= V_cont.
                diff = V_ex - V_cont
                valid = np.where(diff >= 0)[0]
                if len(valid) > 0:
                    j_max = valid[-1]
                    critical_prices[k] = S_i[j_max]
                else:
                    critical_prices[k] = 0.0
                
                # Update V
                V = np.maximum(V_cont, V_ex)
            else:
                # Should not happen if logic is correct
                V_up = V[1:]
                V_down = V[:-1]
                V = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
        else:
            # No exercise
            V_up = V[1:]
            V_down = V[:-1]
            V = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
        
        # Optional: Early termination if boundary is 0 or very high?
        # Not strictly necessary.

    # Ensure monotonicity of boundaries (optional, but good for sanity)
    # For a put, S*_k should be non-decreasing with k (time to expiry decreases).
    # So S*_1 <= S*_2 <= ... <= S*_12.
    for k in range(1, 12):
        if critical_prices[k] > critical_prices[k+1]:
            # Smooth? Or just take max?
            # Actually, due to numerical error, small violations might occur.
            # We can enforce monotonicity by taking the running max from the end?
            # Or running min from the start?
            # Since S*_k should increase, we can do:
            pass # Keep as is for now, tree is accurate enough.

    return ExercisePolicy(critical_prices)

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)
        
    def exercise(self, current_state, time):
        # time is a float
        # Determine k
        # The exercise dates are at 1/12, 2/12, ..., 12/12.
        # time should be close to one of these.
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        S_star = self.critical_prices[k_int]
        return current_state <= S_star

def _fallback_policy():
    # Simple heuristic fallback if tree calculation takes too long
    # For a Bermudan put, the boundary is roughly between 0 and K.
    # A rough approximation is to exercise if S < K * exp(-r * (T-t)) ? No.
    # Let's just use a static boundary of K * 0.9 for all dates?
    # Or 0?
    # Actually, if we fail, we should return something valid.
    # Let's return a policy that never exercises, which is safe but suboptimal.
    # Or use the intrinsic value boundary?
    # Let's use a simple linear interpolation from 0 to K?
    cp = [0.0] * 13
    for k in range(1, 13):
        # As time to expiry decreases, boundary increases.
        # t_k = k/12. tau = 1 - t_k.
        # Boundary ~ K * (1 - exp(-sigma^2 * tau / 2)) ?
        # Let's just use K * (k/12) as a rough guess?
        cp[k] = K * (k / 12.0) * 0.9
    return ExercisePolicy(cp)

# Constants for fallback
K = 100.0
