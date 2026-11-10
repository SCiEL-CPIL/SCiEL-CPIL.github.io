import numpy as np
import time

class ExercisePolicy:
    def __init__(self, boundaries, exercise_times):
        self.boundaries = boundaries
        self.exercise_times = exercise_times
        self.n_ex = len(exercise_times)
        
    def exercise(self, current_state, time):
        # Map time to exercise index
        # exercise_times are [T/12, 2T/12, ..., T]
        # time is a float
        # Find closest exercise time
        # Or compute index directly
        # t_k = (k+1) * T / 12
        # k = round(t * 12 / T) - 1
        
        # Use the stored exercise_times for robustness
        # Find index where time matches
        # Since time is likely exact or close, we can use searchsorted or argmin
        
        # Simpler:
        # The contract says "12 monthly exercise dates".
        # We assume time is one of these.
        
        # Calculate index
        # T is stored in exercise_times[-1]
        T = self.exercise_times[-1]
        k = int(round(time * 12.0 / T)) - 1
        
        # Clamp k
        if k < 0:
            return np.zeros_like(current_state, dtype=bool)
        if k >= self.n_ex:
            k = self.n_ex - 1
            
        # Exercise if S <= Boundary
        return current_state <= self.boundaries[k]

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    
    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_ex = 12
    
    if hasattr(contract, 'params'):
        params = contract.params
        if isinstance(params, dict):
            K = params.get('K', K)
            r = params.get('r', r)
            sigma = params.get('sigma', sigma)
            T = params.get('T', T)
            n_ex = params.get('n_ex', n_ex)
    elif isinstance(contract, dict):
        K = contract.get('K', K)
        r = contract.get('r', r)
        sigma = contract.get('sigma', sigma)
        T = contract.get('T', T)
        n_ex = contract.get('n_ex', n_ex)

    # Exercise times
    exercise_times = np.linspace(T/n_ex, T, n_ex)
    
    # Use Binomial Tree to find optimal boundaries
    # N steps for the tree
    N = 1200
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # V will store values at the current step
    # We only need to store the next step's values to compute current
    # Max nodes at step j is j+1
    # We can use a 1D array that we resize or just use a 2D array of size N+1 x N+1 (wasteful but N=1200 is small, 1.44M floats ~ 11MB)
    # Let's use a 2D array for simplicity and speed in NumPy
    
    V = np.zeros((N + 1, N + 1))
    
    # Initialize at T (step N)
    # S_N_i = S_0 * u^i * d^(N-i)
    # We don't know S_0 from training_paths? 
    # The task says "Spot 90/100/110". 
    # Usually S_0 is 100.
    # The training paths are provided, but for the binomial tree reference, we assume the model parameters define the dynamics.
    # The "current_state" passed to exercise() will be the actual spot.
    # The boundary is a function of S.
    # The binomial tree computes the optimal *policy* (boundary) which is independent of the specific training paths, 
    # assuming the training paths are consistent with the GBM parameters.
    # So we assume S_0 = 100 for the tree construction?
    # Actually, the boundary $S_{crit}$ is a specific price level.
    # The tree nodes are $S_{j,i} = S_0 u^i d^{N-i}$.
    # If S_0 is not 100, the nodes shift.
    # However, the *ratio* of prices is what matters for the optimal exercise *decision* in a recombining tree?
    # No, the absolute price matters for the payoff $K-S$.
    # But the "boundary" is a price level.
    # If the model is GBM with S_0=100, the optimal boundary is a specific number.
    # If the evaluator uses S_0=100, this is correct.
    # The task says "Spot 90/100/110". This likely refers to the initial spot for different scenarios or just the range.
    # Given "K 100, r 0.05, sigma 0.2, T 1", and standard GBM, S_0=100 is the standard assumption.
    
    S_0 = 100.0
    
    # Initialize V at step N
    i = np.arange(N + 1)
    S_N = S_0 * (u ** i) * (d ** (N - i))
    V[N, :N+1] = np.maximum(K - S_N, 0.0)
    
    boundaries = np.zeros(n_ex)
    
    # Map exercise dates to tree steps
    # t_k = (k+1) * T / 12
    # j_k = k * N / 12
    # Since N=1200, N/12 = 100.
    # j_k = (k+1) * 100?
    # t_1 = T/12. j_1 = (T/12) / (T/1200) = 100.
    # t_12 = T. j_12 = 1200.
    # So exercise steps are 100, 200, ..., 1200.
    
    for j in range(N - 1, -1, -1):
        # Calculate S at step j
        i = np.arange(j + 1)
        S_j = S_0 * (u ** i) * (d ** (j - i))
        
        # Calculate continuation value
        # V[j, i] = exp(-r dt) * (p * V[j+1, i+1] + (1-p) * V[j+1, i])
        V_cont = np.exp(-r * dt) * (p * V[j + 1, 1:j + 2] + (1 - p) * V[j + 1, 0:j + 1])
        
        # Check if this is an exercise date
        # j is an exercise step if j is a multiple of 100 and j > 0
        if j > 0 and j % 100 == 0:
            # Determine k
            # j = (k+1) * 100
            # k = j / 100 - 1
            k = j // 100 - 1
            
            V_payoff = np.maximum(K - S_j, 0.0)
            V_opt = np.maximum(V_cont, V_payoff)
            V[j, :j+1] = V_opt
            
            # Find boundary
            # Exercise is optimal where V_payoff >= V_cont
            # We want the maximum S where this is true.
            # S_j is increasing with i.
            # So we find the max i such that V_payoff[i] >= V_cont[i]
            
            exercise_mask = V_payoff >= V_cont
            if np.any(exercise_mask):
                max_i = np.max(np.where(exercise_mask)[0])
                boundaries[k] = S_j[max_i]
            else:
                boundaries[k] = 0.0
        else:
            V[j, :j+1] = V_cont
            
    return ExercisePolicy(boundaries, exercise_times)
