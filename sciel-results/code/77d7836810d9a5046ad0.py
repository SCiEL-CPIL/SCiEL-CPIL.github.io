import numpy as np

class ExercisePolicy:
    def __init__(self, decisions, S0, r, sigma, T, n_tree_steps, exercise_indices):
        self.decisions = decisions # 2D boolean array [n_tree_steps+1, n_tree_steps+1]
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_tree_steps
        self.dt = T / n_tree_steps
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.exercise_indices = exercise_indices
        self.exercise_set = set(exercise_indices)
        
    def exercise(self, current_state, time):
        # Map time to tree index
        i = int(round(time / self.dt))
        i = max(0, min(self.n_steps, i))
        
        if i not in self.exercise_set:
            return np.zeros(len(current_state), dtype=bool)
            
        S_curr = np.asarray(current_state)
        S0_d_i = self.S0 * (self.d ** i)
        
        # Calculate node index j
        # S_{i,j} = S0 * u^j * d^(i-j)
        # j = log(S / (S0 * d^i)) / log(u/d)
        log_ratio = np.log(S_curr / S0_d_i)
        log_ud = np.log(self.u / self.d)
        j_float = log_ratio / log_ud
        j_int = np.round(j_float).astype(int)
        j_int = np.clip(j_int, 0, i)
        
        return self.decisions[i, j_int]

def fit_exercise_policy(training_paths, contract):
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_tree_steps = 1200
    dt = T / n_tree_steps
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Exercise at t = k/12 for k=1..12
    # Tree index i = k * (n_tree_steps / 12)
    exercise_indices = [k * (n_tree_steps // 12) for k in range(1, 13)]
    exercise_indices[-1] = n_tree_steps
    
    V = np.zeros((n_tree_steps + 1, n_tree_steps + 1))
    decisions = np.zeros((n_tree_steps + 1, n_tree_steps + 1), dtype=bool)
    
    S0 = 100.0
    
    for i in range(n_tree_steps, -1, -1):
        js = np.arange(i + 1)
        S_vals = S0 * (u ** js) * (d ** (i - js))
        payoff = np.maximum(K - S_vals, 0.0)
        
        if i == n_tree_steps:
            V[i, :i+1] = payoff
            decisions[i, :i+1] = payoff > 0.0
        else:
            V_up = V[i+1, 1:i+2]
            V_down = V[i+1, 0:i+1]
            V_cont = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
            
            if i in exercise_indices:
                V_opt = np.maximum(payoff, V_cont)
                V[i, :i+1] = V_opt
                decisions[i, :i+1] = (payoff >= V_cont)
            else:
                V[i, :i+1] = V_cont
                decisions[i, :i+1] = False
                
    return ExercisePolicy(decisions, S0, r, sigma, T, n_tree_steps, exercise_indices)
