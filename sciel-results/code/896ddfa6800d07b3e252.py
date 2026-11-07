"""Market making with inventory risk - HJB-based optimal quoting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed optimal depths for each regime
# Key: (sigma, A, k, gamma) -> dict with 'bid_depths' and 'ask_depths' arrays
# indexed by [time_step, inventory_index] where inventory_index = q + Q

_REGIME_CACHE = {}

def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths using HJB / first-order condition.
    
    The optimal depth d* satisfies the marginal condition:
    d* = (1/k) * ln( A * k * (1 + gamma * sigma^2 * |q| / (A * exp(-k*d*) * dt) ) )
    
    We solve this iteratively. The key insight is that with quadratic inventory
    cost (gamma/2)*sigma^2*q^2*dt per step, the optimal quoting shifts:
    - When q > 0 (long inventory), we want to sell more -> narrower ask, wider bid
    - When q < 0 (short inventory), we want to buy more -> narrower bid, wider ask
    
    Using the HJB approach:
    V(t, q) = max over bid_depth, ask_depth of:
      - (gamma/2)*sigma^2*q^2*dt
      + p_bid * (bid_depth - sigma^2*dt*q)  [expected value from bid fill]
      + p_ask * (-ask_depth - sigma^2*dt*q)  [expected value from ask fill]
      + (1-p_bid-p_ask) * V_next  [no fill]
    
    where p_bid = 1 - exp(-A*exp(-k*bid_depth)*dt)
          p_ask = 1 - exp(-A*exp(-k*ask_depth)*dt)
    
    The first-order condition for bid_depth (when q > 0, we widen bid to reduce inventory):
    d_bid* = (1/k) * ln( A * k * dt / (1 + gamma*sigma^2*q*dt / (1 - exp(-A*exp(-k*d)*dt))) )
    
    For a cleaner approach, we use the fact that the optimal spread adjusts
    linearly with inventory in the small-volatility limit:
    d_bid = d0 + alpha * max(0, q)
    d_ask = d0 + alpha * max(0, -q)
    
    where d0 = 1/k is the risk-neutral depth and alpha captures the inventory
    risk adjustment. We compute alpha by solving the HJB at representative
    inventory levels.
    """
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # Base depth (risk-neutral)
    d0 = 1.0 / k
    
    # Compute optimal depths for each (time, inventory)
    # We work backwards from t=T (no more quoting needed, but we still quote at each step)
    # Actually, at each step we quote and then observe fills. The value function:
    # V[n, q] = max_{db, da} { -(gamma/2)*sigma^2*q^2*dt 
    #              + p_b*(db - sigma^2*dt*q + V[n+1, q+1]) 
    #              + p_a*(-da - sigma^2*dt*q + V[n+1, q-1])
    #              + (1-p_b-p_a)*(V[n+1, q]) }
    # where p_b = 1-exp(-A*exp(-k*db)*dt), p_a = 1-exp(-A*exp(-k*da)*dt)
    # V[steps, q] = 0 (no terminal charge)
    
    # For efficiency, we'll compute the optimal depths using a grid of candidate depths
    # and pick the best one for each (time, inventory) state.
    
    # Candidate depths: logarithmically spaced from very small to very large
    # d ranges from ~0.001 to ~10/k
    max_d = 10.0 / k
    min_d = 0.001
    n_candidates = 200
    candidate_depths = np.geomspace(min_d, max_d, n_candidates)
    
    # Precompute fill probabilities for each candidate depth
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * candidate_depths)  # (n_candidates,)
    fill_probs = 1.0 - np.exp(-A * exp_kd * dt)  # (n_candidates,)
    
    # Value function: V[n, q] for n in 0..steps-1, q in -Q..Q
    # V[n, q] is the optimal expected profit from step n onwards given inventory q
    # We only need V for the current step to compute the next step's optimal action,
    # but we need V[n+1] to compute V[n].
    
    # To save memory, we only store the current and next value functions
    V_next = np.zeros(n_inv)  # V at time steps (terminal)
    
    # Store optimal depths: bid_depths[n, q_idx], ask_depths[n, q_idx]
    bid_depths = np.empty((n_steps, n_inv))
    ask_depths = np.empty((n_steps, n_inv))
    
    # Time budget: we can't spend too much time computing
    # With 1000 steps, 61 inventory levels, and 200 candidates, this is 1000*61*200 = 12.2M ops per side
    # That's manageable.
    
    inv_range = np.arange(-Q, Q + 1)  # -Q, -Q+1, ..., Q
    
    for n in range(n_steps - 1, -1, -1):
        V_curr = np.empty(n_inv)
        
        for qi in range(n_inv):
            q = inv_range[qi]
            
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q >= Q:
                # Cannot have bid fill (already at max long inventory)
                # Only ask side is active
                best_val = -np.inf
                best_da = candidate_depths[0]
                
                for ci in range(n_candidates):
                    da = candidate_depths[ci]
                    pa = fill_probs[ci]
                    
                    # If ask fill: inventory decreases by 1, earn -da
                    # If no fill: no change
                    val = -inv_cost + pa * (-da + V_next[qi - 1]) + (1.0 - pa) * V_next[qi]
                    
                    if val > best_val:
                        best_val = val
                        best_da = da
                
                V_curr[qi] = best_val
                bid_depths[n, qi] = np.inf  # No bid quote
                ask_depths[n, qi] = best_da
                
            elif q <= -Q:
                # Cannot have ask fill (already at max short inventory)
                # Only bid side is active
                best_val = -np.inf
                best_db = candidate_depths[0]
                
                for ci in range(n_candidates):
                    db = candidate_depths[ci]
                    pb = fill_probs[ci]
                    
                    # If bid fill: inventory increases by 1, earn db
                    val = -inv_cost + pb * (db + V_next[qi + 1]) + (1.0 - pb) * V_next[qi]
                    
                    if val > best_val:
                        best_val = val
                        best_db = db
                
                V_curr[qi] = best_val
                bid_depths[n, qi] = best_db
                ask_depths[n, qi] = np.inf  # No ask quote
                
            else:
                # Both sides active
                # We need to find optimal (db, da) jointly
                # This is a 2D optimization. For efficiency, we can fix one and optimize the other,
                # or use a coarse grid.
                
                # Use a reduced candidate set for the 2D search
                # Or: exploit the structure. The optimal db and da are somewhat independent
                # in the sense that db affects bid fill prob and da affects ask fill prob.
                # Let's do a coarse 2D grid.
                
                n_coarse = 50  # 50x50 = 2500 evaluations per state
                coarse_indices = np.linspace(0, n_candidates - 1, n_coarse, dtype=int)
                
                best_val = -np.inf
                best_db = candidate_depths[0]
                best_da = candidate_depths[0]
                
                for ci_b in coarse_indices:
                    db = candidate_depths[ci_b]
                    pb = fill_probs[ci_b]
                    
                    for ci_a in coarse_indices:
                        da = candidate_depths[ci_a]
                        pa = fill_probs[ci_a]
                        
                        # Both fills can happen independently
                        # p_both = pb * pa
                        # p_bid_only = pb * (1 - pa)
                        # p_ask_only = (1 - pb) * pa
                        # p_none = (1 - pb) * (1 - pa)
                        
                        # When both fill: net inventory change is 0, net earnings = db - da
                        val = (-inv_cost
                               + pb * pa * (db - da + V_next[qi])
                               + pb * (1.0 - pa) * (db + V_next[qi + 1])
                               + (1.0 - pb) * pa * (-da + V_next[qi - 1])
                               + (1.0 - pb) * (1.0 - pa) * V_next[qi])
                        
                        if val > best_val:
                            best_val = val
                            best_db = db
                            best_da = da
                
                V_curr[qi] = best_val
                bid_depths[n, qi] = best_db
                ask_depths[n, qi] = best_da
        
        V_next = V_curr
    
    return bid_depths, ask_depths


def _get_optimal_depths(params):
    """Get or compute optimal depths for given parameters."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _REGIME_CACHE:
        start_time = time.time()
        bid_depths, ask_depths = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)
        _REGIME_CACHE[key] = (bid_depths, ask_depths)
        
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > 200:
            # We're running out of time, but we've already computed one regime
            # Subsequent regimes will use a simpler heuristic
            pass
    
    return _REGIME_CACHE[key]


def _simple_heuristic_depths(inventory, params, t, steps, T):
    """Fallback simple heuristic if DP is too slow."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    dt = T / steps
    
    n = len(inventory)
    
    # Base depth
    d0 = 1.0 / k
    
    # Inventory adjustment: shift depth based on inventory
    # When long (q > 0), widen bid (to sell less aggressively) and narrow ask
    # When short (q < 0), widen ask and narrow bid
    
    # The adjustment is proportional to inventory and the risk parameter
    # alpha = gamma * sigma^2 * dt / (A * k * exp(-k*d0)) approximately
    # Simplified: alpha ~ gamma * sigma^2 * |q| * dt
    
    q = inventory.astype(float)
    
    # Time decay: as we approach the end, we become more aggressive (narrower quotes)
    # to close out inventory
    time_factor = 1.0 + 0.5 * (1.0 - t / T)  # 1.5 at start, 1.0 at end
    
    # Inventory risk adjustment
    # d_bid = d0 * time_factor + beta * max(0, q)  (wider bid when long)
    # d_ask = d0 * time_factor + beta * max(0, -q)  (wider ask when short)
    beta = gamma * sigma**2 * dt / (2.0 * k)
    
    bid_depths = d0 * time_factor + beta * np.maximum(0, q)
    ask_depths = d0 * time_factor + beta * np.maximum(0, -q)
    
    # Ensure minimum depth
    min_depth = 0.001
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # At inventory limits, don't quote on the side that would increase inventory
    bid_depths[np.array(inventory) >= Q] = np.inf
    ask_depths[np.array(inventory) <= -Q] = np.inf
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    """
    n = len(inventory)
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Check time budget
    # We need to be fast here - this is called 1000 times per regime
    # Total budget: 330s for all evaluations
    
    # Try to use pre-computed optimal depths
    try:
        key = (sigma, A, k, gamma, T, steps, Q)
        if key in _REGIME_CACHE:
            bid_depths_table, ask_depths_table = _REGIME_CACHE[key]
        else:
            # Compute if we have time
            elapsed_check = time.time()
            bid_depths_table, ask_depths_table = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)
            _REGIME_CACHE[key] = (bid_depths_table, ask_depths_table)
    except Exception:
        # Fallback to heuristic
        return _simple_heuristic_depths(inventory, params, t, steps, T)
    
    # Determine current time step
    dt = T / steps
    n_step = int(round(t / dt))
    n_step = np.clip(n_step, 0, steps - 1)
    
    # Map inventory to index: q -> q + Q
    inv_indices = np.array(inventory) + Q
    inv_indices = np.clip(inv_indices, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_depths_table[n_step, inv_indices].astype(float)
    ask_depths = ask_depths_table[n_step, inv_indices].astype(float)
    
    # Ensure validity: values must be > -3/k, no NaN, no -inf
    min_valid = -3.0 / k
    
    # Replace invalid values
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, np.inf)
    
    # Clip to valid range (depths should be positive)
    bid_depths = np.where(bid_depths > min_valid, bid_depths, np.inf)
    ask_depths = np.where(ask_depths > min_valid, ask_depths, np.inf)
    
    # Ensure correct shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
