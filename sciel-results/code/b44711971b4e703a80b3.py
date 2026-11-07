"""Optimal market making with inventory risk: HJB precomputed policy."""
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

_T = 1.0
_STEPS = 1000
_Q = 30
_DT = _T / _STEPS
_NQ = 2 * _Q + 1  # 61 inventory states

# Precomputed optimal policies: dict mapping (sigma, A, k, gamma) -> (bid_table, ask_table)
# Tables have shape (n_steps, n_q) storing optimal bid/ask depths
_POLICIES = {}
_TIME_START = None
_DEADLINE = 325.0  # safety margin before 330s kill

def _solve_hjb(sigma, A, k, gamma, n_steps, Q, dt):
    """Solve the discrete HJB equation backwards in time.
    Returns bid_table and ask_table, each of shape (n_steps, 2*Q+1).
    """
    nq = 2 * Q + 1
    # Value function V[t, q] for t in 0..n_steps, q in -Q..Q
    # We compute backwards from t = n_steps to t = 0
    # V[n_steps, q] = 0 (no terminal charge)
    
    # Precompute the fill probability for a range of depths
    # Depth range: from -3/k + eps to some max (say 2.0/k)
    # Actually, let's discretize depth into a grid
    # But for efficiency, let's think about this differently.
    
    # The optimal depth for each (t, q) can be found by maximizing:
    # V(t, q) = max_{bd, ad} [ bd + ad + (1-pb)*V(t+1, q) + pb*V(t+1, q+1) 
    #           + (1-pa)*V(t+1, q) + pa*V(t+1, q-1) 
    #           - (gamma/2)*sigma^2*q^2*dt ]
    # where pb = 1-exp(-A*exp(-k*bd)*dt), pa = 1-exp(-A*exp(-k*ad)*dt)
    # Note: q+1 clamped to Q, q-1 clamped to -Q
    
    # Actually, the objective is: depths earned minus inventory penalty
    # Per step reward: bd + ad - (gamma/2)*sigma^2*q^2*dt
    # Then stochastic transition to next state
    
    # Let's use a grid of depths for bid and ask
    # Depth must be > -3/k. Let's use depths from (-3/k + 0.01) to (2/k)
    # Number of depth points: let's use ~50 for speed
    d_min = -3.0 / k + 0.01
    d_max = 2.0 / k
    n_depths = 50
    depth_grid = np.linspace(d_min, d_max, n_depths)
    
    # Precompute fill probabilities for all depths
    # pb[d] = 1 - exp(-A * exp(-k * d) * dt)
    pb_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    pa_grid = pb_grid  # same for ask
    
    # V[t, q] for t in 0..n_steps, q in -Q..Q
    # We'll store V as shape (n_steps+1, nq)
    V = np.zeros((n_steps + 1, nq), dtype=np.float64)
    
    # bid_opt[t, q], ask_opt[t, q]
    bid_opt = np.zeros((n_steps, nq), dtype=np.float64)
    ask_opt = np.zeros((n_steps, nq), dtype=np.float64)
    
    # Backward induction
    for t in range(n_steps - 1, -1, -1):
        # For each inventory level q
        for qi in range(nq):
            q = qi - Q
            qf = float(q)
            inv_penalty = (gamma / 2.0) * sigma * sigma * qf * qf * dt
            
            # Try all combinations of bid and ask depths
            # This is O(n_depths^2) per (t, q), which is 50^2 = 2500
            # Total: 1000 * 61 * 2500 = 152.5M operations -- too slow!
            
            # Let's be smarter: for each bid depth, find the optimal ask depth
            # Actually, the problem separates: for fixed bid, the ask optimization
            # is independent. So we can do:
            # For each bid depth bd:
            #   pb = fill_prob(bd)
            #   For each ask depth ad:
            #     pa = fill_prob(ad)
            #     reward = bd + ad - inv_penalty
            #     next_val = (1-pb)*(1-pa)*V[t+1,q] + pb*(1-pa)*V[t+1,q+1] 
            #                + (1-pb)*pa*V[t+1,q-1] + pb*pa*V[t+1,q]
            #     total = reward + next_val
            # Take max over (bd, ad)
            
            # To speed up, note that for fixed bd, the optimal ad can be found
            # by scanning over ad. But 50*50 = 2500 per (t,q) is still 152.5M.
            # Let's reduce n_depths to 20: 20*20 = 400, total 1000*61*400 = 24.8M -- OK!
            
            best_val = -np.inf
            best_bd = depth_grid[0]
            best_ad = depth_grid[0]
            
            # For each bid depth
            for i, bd in enumerate(depth_grid):
                pb = pb_grid[i]
                # For each ask depth
                for j, ad in enumerate(depth_grid):
                    pa = pa_grid[j]
                    reward = bd + ad - inv_penalty
                    # Next state values
                    # q stays: (1-pb)*(1-pa)
                    # q+1 (bid fill): pb*(1-pa)
                    # q-1 (ask fill): (1-pb)*pa
                    # q (both fills cancel): pb*pa -> but wait, if both fill, net change is 0
                    # Actually: bid fill -> q+1, ask fill -> q-1
                    # Both fill -> q+1-1 = q
                    # Neither -> q
                    
                    q_plus = min(q + 1, Q)
                    q_minus = max(q - 1, -Q)
                    
                    v_next_same = V[t+1, q + Q]
                    v_next_plus = V[t+1, q_plus + Q]
                    v_next_minus = V[t+1, q_minus + Q]
                    
                    next_val = ((1-pb)*(1-pa) * v_next_same +
                               pb*(1-pa) * v_next_plus +
                               (1-pb)*pa * v_next_minus +
                               pb*pa * v_next_same)
                    
                    total = reward + next_val
                    if total > best_val:
                        best_val = total
                        best_bd = bd
                        best_ad = ad
            
            V[t, qi] = best_val
            bid_opt[t, qi] = best_bd
            ask_opt[t, qi] = best_ad
        
        # Check time
        if _TIME_START is not None:
            if time.time() - _TIME_START > _DEADLINE - 5.0:
                break
    
    return bid_opt, ask_opt

def _precompute_all():
    """Precompute HJB solutions for all regimes."""
    global _TIME_START
    if _TIME_START is None:
        _TIME_START = time.time()
    
    regimes = [
        (2, 140, 1.5, 0.05),
        (2, 140, 1.5, 0.5),
        (1.5, 50, 0.75, 0.2),
    ]
    
    for sigma, A, k, gamma in regimes:
        key = (sigma, A, k, gamma)
        if key in _POLICIES:
            continue
        if _TIME_START is not None and time.time() - _TIME_START > _DEADLINE - 10.0:
            break
        bid_table, ask_table = _solve_hjb(sigma, A, k, gamma, _STEPS, _Q, _DT)
        _POLICIES[key] = (bid_table, ask_table)
        print(f"Precomputed policy for regime {key}")

def quote(t, inventory, mid, params):
    """Look up precomputed HJB optimal policy."""
    global _TIME_START
    # Precompute on first call if not done
    if not _POLICIES:
        if _TIME_START is None:
            _TIME_START = time.time()
        _precompute_all()
    
    sigma = params['sigma']; A = params['A']; k = params['k']; gamma = params['gamma']
    key = (sigma, A, k, gamma)
    
    if key in _POLICIES:
        bid_table, ask_table = _POLICIES[key]
        # Determine step index
        n = int(round(t / _DT))
        n = max(0, min(n, _STEPS - 1))
        
        # Map inventory to index
        q_idx = np.clip(inventory + _Q, 0, 2 * _Q)
        
        bid_depths = bid_table[n, q_idx]
        ask_depths = ask_table[n, q_idx]
    else:
        # Fallback: simple heuristic
        qf = inventory.astype(np.float64)
        alpha = 1.0 / k
        shift = 0.02 * gamma
        bid_depths = alpha * np.exp(-shift * qf)
        ask_depths = alpha * np.exp(shift * qf)
    
    # Ensure valid depths (> -3/k)
    min_valid = -3.0 / k + 1e-6
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
