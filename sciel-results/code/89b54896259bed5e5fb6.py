"""Market-making with inventory risk: Lookahead approximation strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit

# Cache for precomputed constants per regime
# Key: (sigma, A, k, gamma, T, steps, Q)
_CONST_CACHE = {}

def _get_constants(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _CONST_CACHE:
        dt = T / steps
        # Precompute fill probabilities for a grid of depths
        # We need to evaluate depth in a reasonable range. 
        # Optimal depth is usually around 1/k. 
        # Let's create a grid of depths to map to probabilities.
        # However, the policy returns continuous depths. 
        # We can compute probability on the fly or use a lookup.
        # Given the small number of steps in lookahead (e.g., 5), on-the-fly is fast enough.
        
        # For the value function approximation, we use a simple quadratic form:
        # V(q, t) ≈ -c * q^2 - b(t) * q - a(t)
        # But a simpler approach for the lookahead is to just simulate the expected 
        # cash flow over H steps and add a terminal penalty for inventory.
        
        # Let's store dt and other params
        _CONST_CACHE[key] = {
            'dt': dt,
            'sigma': sigma,
            'A': A,
            'k': k,
            'gamma': gamma,
            'Q': Q,
            'steps': steps
        }
    return _CONST_CACHE[key]

def _simulate_step(inventory, mid, bid_depth, ask_depth, dt, sigma, A, k, Q):
    """
    Simulate one step for a single path.
    Returns: (new_inventory, new_mid, reward)
    Note: This is for the lookahead logic, not the actual execution path which uses vectorization.
    """
    # Probabilities
    p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_depth) * dt)
    p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_depth) * dt)
    
    # Sample outcomes (deterministic expectation for lookahead value estimation? 
    # No, we should estimate the expected value. 
    # E[reward] = p_bid * (-bid_depth) + p_ask * (ask_depth) - (gamma/2)*sigma^2*inventory^2*dt
    # E[new_inventory] = inventory + p_ask - p_bid
    # E[new_mid] = mid (mid is random walk, but for value function we might ignore mid shift 
    # or assume it averages out, or include variance term if value depends on mid. 
    # In this model, mid shift doesn't affect future rewards directly except through 
    # the "distance" which is relative to mid. Since we quote relative to mid, 
    # the absolute mid value doesn't change the optimal relative depths, 
    # so we can ignore mid evolution in the lookahead value estimation for simplicity, 
    # assuming stationarity of the relative problem.)
    
    # Actually, the reward is "depths earned". 
    # If bid is hit, we sell at bid = mid - bid_depth. We receive bid_depth? 
    # "A fill earns its depth". Yes.
    # So if bid hit: +bid_depth. If ask hit: +ask_depth.
    # Inventory change: bid hit -> q+1, ask hit -> q-1.
    
    # Expected immediate reward
    reward = p_bid * bid_depth + p_ask * ask_depth - (0.5 * (1.0/params['gamma'] if False else 1.0) * 0) # Wait, cost is separate.
    
    # The objective is: sum(depths earned) - sum( (gamma/2) sigma^2 q^2 dt )
    # So immediate reward component: p_bid * bid_depth + p_ask * ask_depth
    # Immediate cost component: (gamma/2) * sigma^2 * q^2 * dt
    
    return p_bid, p_ask, reward

def quote(t, inventory, mid, params):
    """
    Lookahead-based quote policy.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback to seed
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    c = _get_constants(sigma, A, k, gamma, T, steps, Q)
    
    # Lookahead horizon
    H = 5
    
    # We want to find bid_depth and ask_depth for each path.
    # Since the problem is symmetric in structure but asymmetric in inventory,
    # we can parameterize:
    # bid_depth = base_depth + tilt * inventory
    # ask_depth = base_depth - tilt * inventory
    # We search for optimal base_depth and tilt for the *current* state distribution?
    # No, we must return per-path depths.
    # However, doing a full optimization per path is too slow.
    # Instead, we can assume the optimal policy is affine in inventory:
    # bid(q) = d0 + m*q
    # ask(q) = d0 - m*q
    # We estimate d0 and m using the lookahead logic on the *average* or *current* inventory distribution?
    # Actually, we can compute the optimal d0 and m for the *entire batch* by assuming 
    # that the optimal affine parameters are the same for all paths (since the model is linear-quadratic).
    # Let's find the best (d0, m) that maximizes the expected lookahead value 
    # averaged over the current inventory distribution.
    
    # Current inventory distribution
    q_mean = np.mean(inventory)
    q_std = np.std(inventory)
    
    # Search space for d0 and m
    # d0 is around 1/k. 
    # m is small.
    # Let's do a small grid search for d0 and m.
    
    # To be fast, we limit the grid.
    d0_grid = np.linspace(0.5/k, 2.0/k, 10)
    m_grid = np.linspace(-0.1, 0.1, 10) # Tilt
    
    best_score = -np.inf
    best_d0 = 1.0/k
    best_m = 0.0
    
    # Precompute costs for grid
    # For each (d0, m), we simulate H steps for a representative set of inventories?
    # Or just use the mean inventory? 
    # The value function is convex, so evaluating at mean is a good approximation 
    # if we also account for variance, but let's start with mean.
    
    # Actually, we can vectorize the lookahead simulation for the grid.
    
    # Let's simplify: 
    # We will compute the expected reward over H steps for a given (d0, m) 
    # starting from the *current* inventory of each path, then average.
    # But doing this for 100x100 grid * 1000 paths * 5 steps is 500k operations, which is fine.
    
    # However, we can be smarter. The optimal d0 and m should be robust.
    # Let's just pick a small grid and evaluate on a subsample of paths to be fast.
    subsample = inventory[::10] # 100 paths
    
    for d0 in d0_grid:
        for m in m_grid:
            # Simulate H steps for subsample
            q = subsample.copy().astype(float)
            total_reward = 0.0
            
            for h in range(H):
                # Depths for each path in subsample
                bid_d = d0 + m * q
                ask_d = d0 - m * q
                
                # Clip depths to be positive and reasonable
                # Minimum depth 0, max maybe 5/k
                bid_d = np.clip(bid_d, 0.0, 5.0/k)
                ask_d = np.clip(ask_d, 0.0, 5.0/k)
                
                # Probabilities
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                
                # Expected reward for this step
                # Reward: p_bid * bid_d + p_ask * ask_d
                # Cost: 0.5 * gamma * sigma**2 * q**2 * dt
                step_reward = np.mean(p_bid * bid_d + p_ask * ask_d - 0.5 * gamma * sigma**2 * q**2 * dt)
                total_reward += step_reward
                
                # Update inventory (expected)
                # E[q_new] = q + p_ask - p_bid
                # Note: p_bid and p_ask are vectors
                q = q + p_ask - p_bid
                
                # Clip inventory to [-Q, Q] for the simulation? 
                # The model says no fill at Q/-Q, but for lookahead we can just let it go 
                # or clip. Clipping is safer for the cost term.
                q = np.clip(q, -Q, Q)
            
            # Terminal cost? "no terminal charge". So we don't add anything at end.
            
            if total_reward > best_score:
                best_score = total_reward
                best_d0 = d0
                best_m = m

    # Apply the found parameters to all paths
    bid_depths = best_d0 + best_m * inventory
    ask_depths = best_d0 - best_m * inventory
    
    # Ensure valid depths
    bid_depths = np.clip(bid_depths, 0.0, 5.0/k)
    ask_depths = np.clip(ask_depths, 0.0, 5.0/k)
    
    # Handle boundaries: no bid fill at Q, no ask fill at -Q
    # If inventory is Q, we shouldn't quote bid (or it won't fill). 
    # Setting depth to inf effectively removes the quote.
    # But the prompt says "np.inf means no quote".
    # If we set depth to 0, we quote at mid. 
    # If we set depth to inf, we don't quote.
    # The model says "no bid fill at q=Q". This implies the probability is 0.
    # It doesn't say we *can't* quote, just that it won't fill.
    # However, quoting at bid_depth > 0 when q=Q yields 0 reward and 0 inventory change.
    # Quoting at bid_depth = 0 yields 0 reward (depth 0) and 0 inventory change.
    # So it doesn't matter much, but to be safe and consistent with "no quote" if it's useless:
    # Actually, if we quote, we pay the "spread" in terms of potential opportunity cost? No, the objective is just depths earned minus inventory cost.
    # If we quote at depth d, and it doesn't fill, we get 0.
    # If we quote at depth 0, and it doesn't fill, we get 0.
    # So we can just leave the depths as calculated.
    
    # However, the constraint "Values below -3/k ... make the whole result invalid"
    # Our clip ensures > 0.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
