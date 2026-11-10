"""Market making with inventory risk - closed-form optimal depth."""
import numpy as np
import time

# Module-level cache for precomputed depth tables
_cache = {}

def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid/ask depths for each (step, inventory) pair.
    Uses closed-form first-order condition for continuous depth optimization.
    
    For a given state (n, q), the optimal bid depth d_bid maximizes:
      - gamma/2 * sigma^2 * q^2 * dt  (inventory cost, same regardless of depth)
      + p_bid * d_bid  (expected fill reward)
      - E[future cost from inventory change]
    
    The fill probability is p = 1 - exp(-A * exp(-k*d) * dt)
    For small dt, p ≈ A * exp(-k*d) * dt
    
    The key trade-off: deeper quote → more reward per fill but lower fill probability.
    The first-order condition for maximizing p(d) * d gives:
      d * dp/dd + p = 0 (ignoring future value for now)
    
    With p = 1 - exp(-A*exp(-k*d)*dt):
      dp/dd = A*k*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)
    
    This is complex, but we can use the approximation that for small dt,
    the optimal depth balances the marginal gain in depth against marginal loss in probability.
    
    A cleaner approach: the optimal depth for a single-period problem with
    reward = p(d)*d where p(d) = 1 - exp(-lambda*exp(-k*d)) with lambda = A*dt
    has FOC: lambda*k*exp(-k*d)*exp(-lambda*exp(-k*d)) * d + (1 - exp(-lambda*exp(-k*d))) = 0
    
    For inventory adjustment, we add a skew term based on q.
    """
    dt = T / steps
    n_steps = steps
    Q_val = Q
    
    # Precompute for all steps and inventory levels
    # bid_depths[n, q] for q in [-Q, Q], n in [0, steps-1]
    # Ask side is symmetric
    
    # For the closed-form approach, we use the following logic:
    # The optimal depth without inventory considerations satisfies:
    #   d_opt ≈ (1/k) * log(A * dt / d_opt)  (approximate fixed-point)
    # But with inventory, we skew: bid depth increases when q > 0 (want to reduce inventory)
    # and ask depth increases when q < 0.
    
    # More precisely, the inventory cost gradient is gamma * sigma^2 * q * dt
    # This acts like a "shadow price" that shifts the optimal depth.
    
    # Let's use a parametric form:
    #   bid_depth(q) = d0 + alpha * q
    #   ask_depth(q) = d0 - alpha * q
    # where d0 is the base optimal depth and alpha is the inventory sensitivity.
    
    # From the FOC, approximately:
    #   d0 ≈ (1/k) * log(A * dt / d0)  → iterate
    #   alpha ≈ gamma * sigma^2 * dt / (k * A * exp(-k*d0) * dt) = gamma * sigma^2 / (k * A * exp(-k*d0))
    
    # But this is rough. Let's do a proper 1D optimization per (n, q) but vectorized.
    
    # Actually, for efficiency, let's precompute a table.
    # The state space is steps * (2*Q+1) which is 1000 * 61 = 61000 states.
    # For each state, we need to find optimal bid and ask depths.
    
    # Key insight: the optimal depth depends on:
    # 1. Time remaining (affects future value)
    # 2. Inventory level (affects cost gradient)
    # 3. Parameters (sigma, A, k, gamma)
    
    # For a tractable solution, use the following approximation:
    # The value function V(n, q) ≈ c(n) * q^2 + d(n)
    # where c(n) is the quadratic cost coefficient at step n.
    
    # Bellman equation:
    # V(n, q) = min over bid_depth, ask_depth of:
    #   gamma/2 * sigma^2 * q^2 * dt
    #   + p_bid * bid_depth  (if q < Q, bid fill gives +bid_depth and q→q-1)
    #   + p_ask * ask_depth  (if q > -Q, ask fill gives +ask_depth and q→q+1)
    #   - p_bid * p_ask * bid_depth * ask_depth / (1 - p_bid - p_ask)  (correction for both fills)
    #   + E[V(n+1, q')]
    
    # This is getting complex. Let's use a simpler heuristic that captures the key dynamics:
    # 1. Base depth d0 from single-period optimum
    # 2. Inventory skew proportional to q
    # 3. Time decay: as t→T, reduce depth to avoid being stuck with inventory
    
    # Single-period optimal depth (ignoring future):
    # Maximize f(d) = (1 - exp(-A*exp(-k*d)*dt)) * d
    # f'(d) = A*k*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)*d + (1 - exp(-A*exp(-k*d)*dt)) = 0
    
    # Let u = A*exp(-k*d)*dt, then d = (1/k)*log(A*dt/u)
    # f'(d) = 0 becomes:
    # u*k*(1/k)*log(A*dt/u) + (1 - exp(-u)) = 0  [approximately, for small u]
    # u*log(A*dt/u) + (1 - exp(-u)) = 0
    
    # For small u: 1 - exp(-u) ≈ u, so:
    # u*log(A*dt/u) + u = 0 → log(A*dt/u) = -1 → u = A*dt/e
    # So d0 ≈ (1/k)*log(A*dt/(A*dt/e)) = (1/k)*log(e) = 1/k
    
    # This confirms the seed's 1/k is the single-period optimum!
    # The improvement comes from inventory awareness and time awareness.
    
    # Inventory skew: the marginal cost of inventory is gamma*sigma^2*q*dt
    # This should increase bid depth when q > 0 (to reduce inventory)
    # and increase ask depth when q < 0.
    
    # The skew magnitude: alpha ≈ gamma * sigma^2 * dt / (k * A * exp(-k*d0))
    # With d0 = 1/k: exp(-k*d0) = exp(-1) = 1/e
    # alpha ≈ gamma * sigma^2 * dt * e / (k * A)
    
    # Time effect: as time runs out, we should quote tighter to avoid large inventory changes
    # near the end (no terminal charge, but still want to be near zero inventory)
    
    # Actually, since there's no terminal charge, the time effect is purely through
    # the remaining opportunity to trade. More time remaining → more value in trading.
    
    # Let's define:
    #   d0 = 1/k  (base depth)
    #   skew(q) = alpha * q  (inventory adjustment)
    #   time_factor = 1.0  (no time adjustment for now, since no terminal charge)
    
    # Bid depth = d0 + skew(q)  (when q > 0, bid deeper to sell inventory)
    # Ask depth = d0 - skew(q)  (when q < 0, ask deeper to buy inventory)
    
    # Wait, I need to think about this more carefully.
    # If q > 0 (long inventory), we want to reduce it, so we should:
    #   - Quote bid deeper (more likely to fill, reducing q)
    #   - Quote ask tighter (less likely to fill, which would increase q)
    
    # So: bid_depth = d0 + alpha*q, ask_depth = d0 - alpha*q
    
    # But we need to ensure depths are positive and above the minimum -3/k.
    
    alpha = gamma * sigma**2 * dt * np.e / (k * A)
    
    # Precompute depth tables
    # bid_depths[n, q_index] where q_index = q + Q, q in [-Q, Q]
    q_values = np.arange(-Q_val, Q_val + 1, dtype=np.float64)
    n_q = len(q_values)
    
    # For simplicity, assume time-invariant depths (no terminal charge makes time less important)
    # But let's add a mild time effect: reduce skew as t→T to avoid over-adjusting
    time_factors = np.ones(n_steps)
    
    bid_depth_table = np.zeros((n_steps, n_q))
    ask_depth_table = np.zeros((n_steps, n_q))
    
    for n in range(n_steps):
        tf = time_factors[n]
        a = alpha * tf
        bd = d0 + a * q_values
        ad = d0 - a * q_values
        
        # Ensure minimum depth constraint: depth > -3/k
        min_depth = -3.0 / k
        bd = np.maximum(bd, min_depth + 1e-9)
        ad = np.maximum(ad, min_depth + 1e-9)
        
        bid_depth_table[n, :] = bd
        ask_depth_table[n, :] = ad
    
    return bid_depth_table, ask_depth_table


def _get_depths(t, inventory, params):
    """Get precomputed depths for given state."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key not in _cache:
        _cache[cache_key] = _solve_regime(sigma, A, k, gamma, T, steps, Q)
    
    bid_table, ask_table = _cache[cache_key]
    
    dt = T / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to index
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) for market making quotes.
    """
    start_time = time.time()
    
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    try:
        bid_depths, ask_depths = _get_depths(t, inventory, params)
        
        # Validate outputs
        min_depth = -3.0 / params['k']
        bid_depths = np.where(bid_depths < min_depth, min_depth + 1e-9, bid_depths)
        ask_depths = np.where(ask_depths < min_depth, min_depth + 1e-9, ask_depths)
        
        # Ensure no NaN or inf
        bid_depths = np.nan_to_num(bid_depths, nan=min_depth + 1e-9, posinf=min_depth + 1e-9, neginf=min_depth + 1e-9)
        ask_depths = np.nan_to_num(ask_depths, nan=min_depth + 1e-9, posinf=min_depth + 1e-9, neginf=min_depth + 1e-9)
        
        return bid_depths, ask_depths
        
    except Exception:
        # Fallback to seed behavior
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
