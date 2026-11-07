"""Optimal market making with inventory risk: Vectorized HJB precomputation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

_REGIMES = [
    {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
    {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
    {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
]

_PRECOMPUTED = {}

def _precompute_regime(regime):
    sigma = regime['sigma']
    A = regime['A']
    k = regime['k']
    gamma = regime['gamma']
    T = regime['T']
    steps = regime['steps']
    Q = regime['Q']
    dt = T / steps
    
    n_q = 2 * Q + 1
    # Depth grid
    # Optimal depth is typically around 1/k to 2/k.
    # Let's use a range [0, 3/k] to be safe.
    max_depth = 3.0 / k
    N_grid = 300  # 300 points for high resolution
    d_grid = np.linspace(0.0, max_depth, N_grid)
    
    # Precompute fill probabilities and rewards for the grid
    exp_kd = np.exp(-k * d_grid)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)
    # Reward if fill: d * p_fill
    # Note: The objective is "depths earned". If fill, earn depth.
    # E[reward] = d * p_fill.
    
    # HJB Backward Induction
    V_next = np.zeros(n_q)
    
    # Tables for optimal depths
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))
    
    # Pre-compute indices for argmax mapping
    # We will use np.argmax on a 3D array (n_q, N_grid, N_grid)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 10:
            # Fallback: fill remaining with a heuristic (e.g., 1/k)
            default_depth = 1.0 / k
            bid_table[n:] = default_depth
            ask_table[n:] = default_depth
            break
            
        # Compute cost for all q
        q_vals = np.arange(-Q, Q + 1)
        cost = 0.5 * gamma * sigma * sigma * q_vals * q_vals * dt
        
        # Compute Delta_+ and Delta_-
        # Delta_+ = V_next[q+1] - V_next[q]
        # Delta_- = V_next[q-1] - V_next[q]
        Delta_plus = np.zeros(n_q)
        Delta_minus = np.zeros(n_q)
        
        # q in [-Q, Q]
        # q+1 is valid if q < Q
        Delta_plus[:n_q-1] = V_next[1:] - V_next[:-1]
        # q-1 is valid if q > -Q
        Delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Boundary handling:
        # If q == Q, no bid fill. p_b should be treated as 0 for transitions.
        # If q == -Q, no ask fill. p_a should be treated as 0 for transitions.
        
        # Vectorized calculation for all q
        # Shapes:
        # V_next: (n_q,)
        # Delta_plus/minus: (n_q,)
        # d_grid: (N_grid,)
        # p_fill: (N_grid,)
        
        # We want J[q, i, j] where i is index for d_b, j is index for d_a
        # J = d_b[i]*p_b[i] + d_a[j]*p_a[j] + V_next[q] 
        #     + p_b[i]*(1-p_a[j]) * Delta_plus[q] 
        #     + (1-p_b[i])*p_a[j] * Delta_minus[q]
        
        # Let's broadcast
        # d_b: (1, N_grid, 1)
        # d_a: (1, 1, N_grid)
        # p_b: (1, N_grid, 1)
        # p_a: (1, 1, N_grid)
        # V_next: (n_q, 1, 1)
        # Delta_plus: (n_q, 1, 1)
        # Delta_minus: (n_q, 1, 1)
        
        db = d_grid[:, np.newaxis, np.newaxis] # (N_grid, 1, 1) -> need (1, N_grid, 1)
        db = d_grid[np.newaxis, :, np.newaxis]
        da = d_grid[np.newaxis, np.newaxis, :]
        
        pb = p_fill[np.newaxis, :, np.newaxis]
        pa = p_fill[np.newaxis, np.newaxis, :]
        
        Vn = V_next[:, np.newaxis, np.newaxis]
        Dp = Delta_plus[:, np.newaxis, np.newaxis]
        Dm = Delta_minus[:, np.newaxis, np.newaxis]
        
        # Base terms
        term_reward = db * pb + da * pa
        term_base = Vn
        
        # Transition terms
        # p_b(1-p_a) Dp + (1-p_b)p_a Dm
        term_trans = pb * (1 - pa) * Dp + (1 - pb) * pa * Dm
        
        J = term_reward + term_base + term_trans
        
        # Apply Boundary Masks
        # q = Q is index n_q - 1
        # q = -Q is index 0
        
        # At q = Q, no bid fill.
        # This means p_b is effectively 0.
        # We can set pb for this row to 0 in the calculation?
        # Or simply mask the J values where i != 0 (if we want d_b=0)?
        # Actually, if p_b=0, the choice of d_b doesn't matter for the value.
        # But we need to return a valid depth.
        # Let's set pb_eff = 0 for q=Q.
        # This will make J independent of i for that row.
        # argmax will pick the first i (index 0, d_b=0).
        
        pb_eff = pb.copy()
        pa_eff = pa.copy()
        
        # Mask for q=Q (last row)
        pb_eff[n_q-1, :, :] = 0.0
        # Mask for q=-Q (first row)
        pa_eff[0, :, :] = 0.0
        
        # Recalculate J with effective probabilities
        # Note: term_reward uses pb/pa. If p=0, reward is 0. Correct.
        # term_trans uses pb/pa. If p=0, terms vanish. Correct.
        
        J_eff = (db * pb_eff + da * pa_eff) + Vn + \
                (pb_eff * (1 - pa_eff) * Dp + (1 - pb_eff) * pa_eff * Dm)
        
        # Find max and argmax
        max_J = J_eff.max(axis=(1, 2))
        argmax_flat = J_eff.argmax(axis=(1, 2))
        
        # Map argmax_flat to i, j
        # argmax_flat is index in flattened (N_grid, N_grid)
        i_idx = argmax_flat // N_grid
        j_idx = argmax_flat % N_grid
        
        bid_table[n, :] = d_grid[i_idx]
        ask_table[n, :] = d_grid[j_idx]
        
        # Update V
        V_curr = max_J - cost
        V_next = V_curr
    
    return bid_table, ask_table

def _init():
    global _PRECOMPUTED
    for r in _REGIMES:
        key = (r['sigma'], r['A'], r['k'], r['gamma'])
        if key not in _PRECOMPUTED:
            _PRECOMPUTED[key] = _precompute_regime(r)

_init()

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    key = (sigma, A, k, gamma)
    
    if key not in _PRECOMPUTED:
        # Fallback
        depth = 1.0 / k
        return np.full(len(inventory), depth), np.full(len(inventory), depth)
        
    bid_table, ask_table = _PRECOMPUTED[key]
    
    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)
    
    qi = inventory + Q
    qi = np.clip(qi, 0, 2 * Q)
    
    bid_depths = bid_table[n, qi].astype(np.float64)
    ask_depths = ask_table[n, qi].astype(np.float64)
    
    # Safety checks
    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, 1.0 / k)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
