# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

# Global state to persist across calls within the same seed run
_acq_state = {
    'call_count': 0,
    'last_mu': None,
    'last_sigma': None,
    'last_f_best': None,
    'best_x_local': None,
    'local_search_active': False,
    'local_search_iter': 0,
    'history_of_improvements': []
}

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 295.0  # Safety margin for 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is safe for division
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # 1. Compute a robust acquisition score for all 2000 candidates
    # We use a hybrid: Expected Improvement (EI) for exploitation and 
    # a scaled uncertainty (LCB-like) for exploration.
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # To balance exploration and exploitation dynamically:
    # Early on, we want to explore. Later, exploit.
    # Since we don't know the exact step count easily without external state,
    # we can use the number of calls to this function as a proxy.
    _acq_state['call_count'] += 1
    calls = _acq_state['call_count']
    
    # Adaptive beta: Start high (exploratory), decay to low (exploitative)
    # Total expected calls is ~25 (30 evals - 5 initial).
    # Let's map calls 1..25 to beta 3.0 .. 0.5
    if calls <= 25:
        beta = 3.0 - (2.5 * (calls - 1) / 24.0)
    else:
        beta = 0.5
        
    lcb = beta * sigma_safe - mu
    
    # Normalize scores to be comparable
    # EI is always >= 0. LCB can be negative.
    # We want higher score = better.
    # Let's combine them. A common trick is to normalize both to [0,1] or mean 0 var 1.
    # However, EI is already a good measure. Let's stick to EI but add a small noise term 
    # or use LCB if EI is flat.
    
    # Actually, a simpler and often more robust approach for "score vector" interfaces
    # where we can do extra work is to perform a local search around the best candidate.
    
    # Compute base scores (EI)
    scores = ei.copy()
    
    # Identify the best candidate index from the global scores
    idx_best = np.argmax(scores)
    
    # 2. Local Search Strategy
    # We have 2000 candidates. We don't have their coordinates.
    # BUT, we have mu and sigma.
    # We cannot run Nelder-Mead without coordinates.
    # WAIT. The task says "No coordinates are passed."
    # This means we CANNOT perform coordinate-based local search like Nelder-Mead 
    # because we don't know where in [-4.5, 4.5]^2 these 2000 points are.
    # 
    # RE-READING TASK: "No coordinates are passed."
    # This is a critical constraint. We only have mu and sigma values.
    # Therefore, any strategy that requires knowing the position of the points 
    # to perform geometric local search is INVALID.
    # 
    # We must select one of the 2000 points based ONLY on their mu, sigma, and f_best.
    # 
    # So, "Multi-Start Deterministic Local Search" in the sense of moving a simplex 
    # in the 2D space is impossible.
    # 
    # However, we can do "Local Search" in the *function space*? No.
    # 
    # Let's reconsider the "Different Strategy".
    # Current: Thompson Sampling over acquisition functions.
    # Inspiration 1 (1.94): Likely a well-tuned static or adaptive EI/LCB.
    # Inspiration 2 (1.33): Thompson Sampling on the objective.
    # Inspiration 3 (1.29): Deterministic TS proxy.
    # 
    # What is a materially different approach that doesn't need coordinates?
    # 
    # Option A: **Bayesian Optimization with a Custom Kernel/Posterior**
    # We can't change the GP.
    # 
    # Option B: **Multi-Objective / Pareto Front**
    # Not applicable, single objective.
    # 
    # Option C: **Confidence Sequence / Sequential Probability Ratio Test**
    # Use the uncertainty to determine when to stop exploring and start exploiting.
    # 
    # Option D: **Random Restart / Diversification**
    # If the search stagnates (f_best doesn't improve for several steps), 
    # force a more exploratory move.
    # 
    # Option E: **Ensemble of Acquisition Functions with Dynamic Weighting based on Recent Performance**
    # The current program does this but with Thompson Sampling (stochastic).
    # A *deterministic* adaptive weighting might be more stable.
    # 
    # Let's look at the "Stagnation" signal.
    # If we detect stagnation, we switch to a highly exploratory mode (e.g., pure noise or max sigma).
    # If we detect improvement, we switch to exploitation (EI or LCB with low beta).
    # 
    # This is a **State-Dependent Acquisition Strategy**.
    # 
    # Strategy:
    # 1. Track history of f_best.
    # 2. If f_best improved in the last K steps, use Exploitation (EI).
    # 3. If f_best did NOT improve in the last K steps, use Exploration (Max Sigma or LCB with high beta).
    # 4. To make it smoother, blend them based on the "stagnation count".
    
    # Initialize state
    if 'f_best_history' not in _acq_state:
        _acq_state['f_best_history'] = []
        
    # Update history
    # Note: We don't know the *new* f_best yet, only the current one passed in.
    # The previous call's f_best is in the history.
    # Actually, `f_best` is the lowest observed value *so far*.
    # So if `f_best` < `previous_f_best`, we had an improvement.
    
    prev_f_best = _acq_state.get('last_f_best', None)
    improved = False
    if prev_f_best is not None:
        if f_best < prev_f_best - 1e-6: # Small tolerance for floating point
            improved = True
            
    # Update stagnation counter
    if 'stagnation_count' not in _acq_state:
        _acq_state['stagnation_count'] = 0
        
    if improved:
        _acq_state['stagnation_count'] = 0
    else:
        _acq_state['stagnation_count'] += 1
        
    _acq_state['last_f_best'] = f_best
    
    stagnation = _acq_state['stagnation_count']
    
    # Determine mode
    # If stagnation is low (0-2), exploit.
    # If stagnation is high (>2), explore.
    
    # Compute Exploitation Score: EI
    s_exploit = ei
    
    # Compute Exploration Score: 
    # We want to go to high uncertainty regions.
    # Max sigma is a good proxy, but it might pick points far from the optimum 
    # even if they are unlikely to be better.
    # A better exploration score is LCB with a very high beta, or simply sigma.
    # Let's use a "Pure Exploration" score: -mu + high_beta * sigma.
    # Or, to be safe, let's use a "Maximally Uncertain" score: sigma.
    # But sigma alone ignores the mean. If mu is very high, even high sigma might not help.
    # LCB with beta=5.0 is a strong explorer.
    
    beta_explore = 5.0
    s_explore = beta_explore * sigma_safe - mu
    
    # Blending
    # If stagnation == 0: 100% Exploit
    # If stagnation == 1: 75% Exploit, 25% Explore
    # If stagnation == 2: 50% Exploit, 50% Explore
    # If stagnation >= 3: 25% Exploit, 75% Explore
    # If stagnation >= 5: 100% Explore
    
    if stagnation <= 0:
        w_exploit = 1.0
    elif stagnation == 1:
        w_exploit = 0.7
    elif stagnation == 2:
        w_exploit = 0.4
    elif stagnation <= 4:
        w_exploit = 0.1
    else:
        w_exploit = 0.0
        
    w_explore = 1.0 - w_exploit
    
    # Normalize scores before blending to ensure they are on similar scales
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
        
    s_exploit_norm = normalize_scores(s_exploit)
    s_explore_norm = normalize_scores(s_explore)
    
    final_scores = w_exploit * s_exploit_norm + w_explore * s_explore_norm
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time check
    if time.time() > deadline:
        # If we are running out of time, just return the exploit scores (faster to compute? No, same cost)
        # Just return what we have.
        pass
        
    return final_scores

# EVOLVE-BLOCK-END
