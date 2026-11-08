"""
Optimization of the second autocorrelation inequality constant C2.
Strategy: Discrete Combinatorial Search via Simulated Annealing.
Instead of optimizing 50 continuous heights, we model the profile as a
piecewise-constant function with K distinct plateau levels.
The optimization variables are the K level values and the assignment of
the 50 bins to these levels.
We use Simulated Annealing to search this mixed discrete/continuous space.
This avoids the local minima traps of smooth surrogates by exploring
the "blocky" structure that often yields high L2/L1 ratios in autocorrelation.
"""
import numpy as np
import time
import random

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 148.0     # safety margin before 150s kill

# Fixed random seeds for reproducibility of the search process itself
RNG = np.random.RandomState(42)
random.seed(42)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # We will search over profiles defined by K levels.
    # Let's try K=4, K=5, K=6.
    # The state is:
    #   levels: array of K positive floats
    #   assignment: array of N integers in [0, K-1]
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
            
    # Helper to generate h from levels and assignment
    def h_from_state(levels, assignment):
        return levels[assignment]

    # Initial diverse starts
    initial_configs = []
    
    # Config 1: Random blocky
    for K in [3, 4, 5, 6]:
        levels = RNG.uniform(0.1, 1.0, K)
        assignment = RNG.randint(0, K, N)
        initial_configs.append((levels.copy(), assignment.copy()))
        
    # Config 2: Symmetric blocky (palindromic assignment)
    for K in [3, 4, 5]:
        levels = RNG.uniform(0.1, 1.0, K)
        half = N // 2
        first_half = RNG.randint(0, K, half)
        assignment = np.concatenate([first_half, first_half[::-1]])
        if N % 2 == 1:
             assignment[half] = RNG.randint(0, K)
        initial_configs.append((levels.copy(), assignment.copy()))
        
    # Config 3: Sorted levels, random assignment
    for K in [4, 5]:
        levels = np.sort(RNG.uniform(0.1, 1.0, K))
        assignment = RNG.randint(0, K, N)
        initial_configs.append((levels.copy(), assignment.copy()))

    # Evaluate initial configs
    for levels, assignment in initial_configs:
        h = h_from_state(levels, assignment)
        track(h)
        
    # Simulated Annealing
    # We will run SA for each K in parallel (sequentially in time, but distinct searches)
    # Actually, let's just run a global SA that allows K to change? 
    # No, changing K is complex. Let's run separate SA loops for different K.
    
    # Time budget allocation
    # We have ~148s.
    # Let's spend 20s on K=4, 20s on K=5, 20s on K=6, and use the rest for refinement of the best found.
    
    sa_configs = [
        (4, 40.0),
        (5, 40.0),
        (6, 40.0)
    ]
    
    # We also want to try K=3 and K=7 briefly
    sa_configs.append((3, 15.0))
    sa_configs.append((7, 15.0))
    
    for K, time_budget in sa_configs:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize SA state
        # Start from the best config for this K if we have one, else random
        # For simplicity, start from a random config for this K
        levels = RNG.uniform(0.1, 1.0, K)
        assignment = RNG.randint(0, K, N)
        
        # Ensure at least one bin uses each level? Not strictly necessary, but helps.
        # Actually, if a level is unused, it's just wasted capacity.
        
        h = h_from_state(levels, assignment)
        current_val = compute_value(h)
        track(h)
        
        if current_val == 0.0:
            current_val = -1.0 # To avoid division issues if we track energy as -val
            
        # SA Parameters
        T_start = 0.1
        T_end = 1e-6
        steps = 50000 # Adjust based on time
        # We estimate time per step. 
        # Let's do a quick timing check for 100 steps
        t0 = time.time()
        for _ in range(100):
            # Dummy step to measure overhead
            pass
        t1 = time.time()
        time_per_step = (t1 - t0) / 100.0
        if time_per_step < 1e-6:
            time_per_step = 1e-6
            
        max_steps = int(time_budget / time_per_step)
        # Cap steps to prevent infinite loop if time_per_step is tiny
        max_steps = min(max_steps, 1000000)
        
        alpha = (T_end / T_start) ** (1.0 / max_steps)
        
        best_sa_val = current_val
        best_sa_levels = levels.copy()
        best_sa_assignment = assignment.copy()
        
        for i in range(max_steps):
            if time.time() > deadline - 10.0:
                break
                
            # 1. Perturb Assignment (Discrete Move)
            # Pick a random bin and change its level
            idx = random.randint(0, N - 1)
            old_level_idx = assignment[idx]
            new_level_idx = random.randint(0, K - 1)
            if old_level_idx == new_level_idx:
                continue
                
            assignment[idx] = new_level_idx
            h_new = h_from_state(levels, assignment)
            val_new = compute_value(h_new)
            
            # 2. Perturb Levels (Continuous Move)
            # Pick a random level and perturb it
            lvl_idx = random.randint(0, K - 1)
            old_level_val = levels[lvl_idx]
            # Perturbation size proportional to current level
            delta = RNG.normal(0, 0.1 * max(0.1, old_level_val))
            new_level_val = max(0.01, old_level_val + delta)
            
            levels[lvl_idx] = new_level_val
            h_new2 = h_from_state(levels, assignment)
            val_new2 = compute_value(h_new2)
            
            # Acceptance Criteria (Metropolis)
            # We want to maximize value. So energy E = -value.
            # dE = E_new - E_old = -val_new - (-val_old) = val_old - val_new
            # Accept if dE < 0 or random < exp(-dE/T)
            
            # Combine the two moves? 
            # Usually SA does one move at a time. 
            # Let's do assignment move first, then level move.
            
            # Move 1: Assignment
            dE1 = current_val - val_new
            if dE1 < 0 or random.random() < np.exp(-dE1 / T_start): # Using T_start as T for now? No, need current T
                pass # Placeholder logic below
            
            # Correct SA Loop
            T = T_start * (alpha ** i)
            
            # Move 1: Assignment
            if random.random() < 0.7: # 70% assignment moves
                idx = random.randint(0, N - 1)
                old_level_idx = assignment[idx]
                new_level_idx = random.randint(0, K - 1)
                if old_level_idx != new_level_idx:
                    assignment[idx] = new_level_idx
                    h_new = h_from_state(levels, assignment)
                    val_new = compute_value(h_new)
                    
                    dE = current_val - val_new
                    if dE < 0 or random.random() < np.exp(-dE / T):
                        current_val = val_new
                        if current_val > best_sa_val:
                            best_sa_val = current_val
                            best_sa_levels = levels.copy()
                            best_sa_assignment = assignment.copy()
                            track(h_new)
                    else:
                        assignment[idx] = old_level_idx # Revert
            else: # 30% level moves
                lvl_idx = random.randint(0, K - 1)
                old_level_val = levels[lvl_idx]
                delta = RNG.normal(0, 0.05 * max(0.1, old_level_val))
                new_level_val = max(0.01, old_level_val + delta)
                
                levels[lvl_idx] = new_level_val
                h_new = h_from_state(levels, assignment)
                val_new = compute_value(h_new)
                
                dE = current_val - val_new
                if dE < 0 or random.random() < np.exp(-dE / T):
                    current_val = val_new
                    if current_val > best_sa_val:
                        best_sa_val = current_val
                        best_sa_levels = levels.copy()
                        best_sa_assignment = assignment.copy()
                        track(h_new)
                else:
                    levels[lvl_idx] = old_level_val # Revert
                    
        # End of SA for this K
        
    # Phase 2: Local Refinement of the Best Found
    # The SA finds a good "blocky" structure. 
    # Now we can relax the constraint that bins must be strictly on levels.
    # We take the best_h found by SA and run a few steps of L-BFGS-B on the exact objective?
    # No, exact objective is not differentiable at the max.
    # But we can use the smooth surrogate from the previous program, starting from the SA result.
    # The SA result is a good global start.
    
    if time.time() < deadline - 5.0:
        # Implement a simple smooth surrogate for refinement
        def _smooth_value(h, beta=100.0):
            h = np.maximum(0.0, h)
            g = np.convolve(h, h)
            m = len(g)
            y = _build_y(g)
            a = y[:m + 1]
            b = y[1:m + 2]
            l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
            norm_1 = np.sum(g) / (m + 1)
            
            g_max = np.max(g)
            if g_max <= 0:
                return 0.0
            lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
            
            if norm_1 <= 0 or lse <= 0:
                return 0.0
            val = l2 / (norm_1 * lse)
            return -val
            
        import scipy.optimize as opt
        
        # Refine best_h
        x0 = best_h.copy()
        bounds = [(0, None)] * N
        
        # Use a few betas
        for beta in [50, 100, 200]:
            if time.time() > deadline - 2.0:
                break
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 0:
                    h_refined = np.maximum(0.0, res.x)
                    track(h_refined)
            except Exception:
                pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
