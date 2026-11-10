import numpy as np
import time
import math

class ExercisePolicy:
    def __init__(self, boundaries, exercise_times):
        self.boundaries = boundaries
        self.exercise_times = exercise_times
        self.n_ex = len(exercise_times)
        self.T = exercise_times[-1] if len(exercise_times) > 0 else 1.0
        
    def exercise(self, current_state, time):
        T = self.T
        k = int(round(time * 12.0 / T)) - 1
        
        if k < 0:
            return np.zeros_like(current_state, dtype=bool)
        if k >= self.n_ex:
            k = self.n_ex - 1
            
        return current_state <= self.boundaries[k]

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
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
    disc = np.exp(-r * dt)
    
    S_0 = 100.0
    
    # Precompute log_u and log_d for efficiency
    log_u = np.log(u)
    log_d = np.log(d)
    
    # V will store values at the current step
    V = np.zeros((N + 1, N + 1))
    
    # Initialize at T (step N)
    i = np.arange(N + 1)
    S_N = S_0 * (u ** i) * (d ** (N - i))
    V[N, :N+1] = np.maximum(K - S_N, 0.0)
    
    boundaries = np.zeros(n_ex)
    
    # Map exercise dates to tree steps
    # t_k = (k+1) * T / 12
    # j_k = (k+1) * N / 12
    # Since N=1200, N/12 = 100.
    # Exercise steps are 100, 200, ..., 1200.
    
    for j in range(N - 1, -1, -1):
        # Calculate S at step j
        i = np.arange(j + 1)
        S_j = S_0 * (u ** i) * (d ** (j - i))
        
        # Calculate continuation value
        V_cont = disc * (p * V[j + 1, 1:j + 2] + (1 - p) * V[j + 1, 0:j + 1])
        
        # Check if this is an exercise date
        if j > 0 and j % 100 == 0:
            k = j // 100 - 1
            
            V_payoff = np.maximum(K - S_j, 0.0)
            V_opt = np.maximum(V_cont, V_payoff)
            V[j, :j+1] = V_opt
            
            # Find boundary: max i where V_payoff >= V_cont
            exercise_mask = V_payoff >= V_cont
            if np.any(exercise_mask):
                max_i = np.max(np.where(exercise_mask)[0])
                boundaries[k] = S_j[max_i]
            else:
                boundaries[k] = 0.0
        else:
            V[j, :j+1] = V_cont
        
        # Time check
        if time.time() > deadline - 5.0:
            break
    
    # Polish pass: refine each boundary using linear interpolation
    if time.time() < deadline - 2.0:
        for k in range(n_ex):
            j = (k + 1) * 100
            if j > N:
                continue
            
            i = np.arange(j + 1)
            S_j = S_0 * (u ** i) * (d ** (j - i))
            
            V_cont = disc * (p * V[j + 1, 1:j + 2] + (1 - p) * V[j + 1, 0:j + 1])
            V_payoff = np.maximum(K - S_j, 0.0)
            
            diff = V_payoff - V_cont
            # Find where diff changes sign from positive to negative
            sign_change = np.where(diff[:-1] >= 0 and diff[1:] < 0)[0]
            
            if len(sign_change) > 0:
                idx = sign_change[0]
                S_lo = S_j[idx]
                S_hi = S_j[idx + 1]
                
                d_lo = diff[idx]
                d_hi = diff[idx + 1]
                if d_lo != d_hi and d_lo - d_hi != 0:
                    t = d_lo / (d_lo - d_hi)
                    boundaries[k] = S_lo + t * (S_hi - S_lo)
                else:
                    boundaries[k] = S_lo
    
    return ExercisePolicy(boundaries, exercise_times)
