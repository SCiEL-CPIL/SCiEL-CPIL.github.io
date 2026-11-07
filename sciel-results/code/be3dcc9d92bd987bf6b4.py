"""Market-making with inventory risk: Direct Monte-Carlo Rollout Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

# Cache for optimized parameters per regime
# Key: (sigma, A, k, gamma, T, steps, Q)
_OPT_CACHE = {}

def _optimize_regime_mc(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute optimal depth parameters (d0, m) using Monte Carlo rollout.
    Policy:
    bid_depth = max(0, d0 + m*q)
    ask_depth = max(0, d0 - m*q)
    
    We simulate N=200 paths of length 1000 for a grid of (d0, m).
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _OPT_CACHE:
        return _OPT_CACHE[key]

    dt = T / steps
    N = 200  # Number of Monte Carlo paths
    
    # Initialize inventory to 0 for all paths
    # The optimal policy should be robust to initial conditions, but 0 is standard.
    # Actually, the evaluator calls quote with whatever inventory the simulation has.
    # Since we don't know the starting inventory of the evaluator's 4000 paths,
    # we assume the policy must work well for the typical distribution.
    # Starting at 0 is the most neutral assumption for finding a global affine policy.
    
    # Grid for d0 and m
    # d0: base spread. Typically around 1/k.
    # m: skew. Reduces risk.
    
    # Coarse Grid
    d0_coarse = np.linspace(0.2 / k, 3.0 / k, 15)
    m_coarse = np.linspace(-0.05, 0.05, 15)
    
    best_score = -np.inf
    best_d0 = 1.0 / k
    best_m = 0.0
    
    # Pre-generate random numbers for the rollout to ensure consistency?
    # No, just use np.random. The variance of the estimate is reduced by N=200.
    # We will generate fresh random numbers for each (d0, m) pair? 
    # That would be noisy. Better to use a fixed set of random numbers for the mid path 
    # and fill decisions?
    # Fill decisions depend on depth, which depends on q, which depends on previous fills.
    # So we must simulate sequentially.
    
    # To speed up, we can vectorize over the (d0, m) grid?
    # No, the trajectories diverge.
    # So we must loop over the grid points.
    # 15x15 = 225 points.
    # Each point: 200 paths * 1000 steps.
    # 225 * 200 * 1000 = 45,000,000 steps.
    # In Python, this is too slow if done with pure Python loops inside the step.
    # We need to vectorize over the 200 paths for each step.
    
    # Let's estimate time:
    # 225 * 1000 = 225,000 vectorized steps of size 200.
    # Each step involves exp, random, etc.
    # 225,000 * ~10 operations = 2.25M operations.
    # NumPy can do ~100M simple ops/sec. So ~0.02s?
    # Wait, 200 is small. Overhead might dominate.
    # Let's try N=500 to reduce variance, but keep grid small.
    
    # Let's refine:
    # 1. Coarse search with N=100, Grid 10x10 (100 points).
    # 2. Fine search with N=500, Grid 10x10 (100 points) around best.
    
    N_coarse = 100
    d0_grid = np.linspace(0.2 / k, 2.5 / k, 10)
    m_grid = np.linspace(-0.03, 0.03, 10)
    
    # We will simulate the mean objective over N paths.
    
    for d0 in d0_grid:
        for m in m_grid:
            # Initialize state for N paths
            q = np.zeros(N)
            objective = 0.0
            
            # Pre-compute mid path increments? No, mid is exogenous.
            # We need consistent mid path for fair comparison?
            # Ideally, yes. But generating 1000 * 225 * N random numbers is a lot.
            # Let's generate one mid path and reuse it?
            # The mid path affects the reward (depths are relative to mid, so reward is just depth).
            # Wait, the reward is "depths earned". It does NOT depend on mid value.
            # The mid value only affects the *price* at which we trade, but the objective 
            # is defined as "depths earned" minus risk.
            # "A fill earns its depth". So reward = depth.
            # This means the mid path is IRRELEVANT for the objective!
            # The only randomness is the FILL events.
            # Fill probabilities depend on depth.
            # Depth depends on q.
            # q depends on fills.
            
            # So we only need to simulate the inventory process q_t.
            # The mid process is irrelevant for the objective calculation.
            # This is a huge simplification.
            
            for step in range(steps):
                # Calculate depths
                bid_d = np.maximum(0, d0 + m * q)
                ask_d = np.maximum(0, d0 - m * q)
                
                # Handle boundaries
                # "no bid fill at q = Q"
                # If q == Q, p_bid = 0
                # If q == -Q, p_ask = 0
                mask_bid = q < Q
                mask_ask = q > -Q
                
                # Calculate probabilities
                # p = 1 - exp(-A * exp(-k * depth) * dt)
                # If depth is 0, p = 1 - exp(-A*dt)
                
                # Use log-sum-exp stability? No, values are small.
                # exp(-k * depth) is safe.
                
                p_bid_raw = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                p_ask_raw = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                
                p_bid = np.where(mask_bid, p_bid_raw, 0.0)
                p_ask = np.where(mask_ask, p_ask_raw, 0.0)
                
                # Reward
                # Sum of (depth * fill_indicator) - risk
                # E[reward] = p_bid * bid_d + p_ask * ask_d - 0.5 * gamma * sigma^2 * q^2 * dt
                
                # We need to SIMULATE the fills to get the next q.
                # u_bid ~ Bernoulli(p_bid)
                # u_ask ~ Bernoulli(p_ask)
                
                u_bid = np.random.rand(N) < p_bid
                u_ask = np.random.rand(N) < p_ask
                
                # Accumulate objective
                # Note: we use the EXPECTED reward for the objective?
                # No, the objective is a random variable.
                # We want to maximize the EXPECTED objective.
                # So we can just use the expected reward term in the accumulation,
                # and use the simulated q for the dynamics.
                # This reduces variance in the objective estimate.
                
                exp_reward = np.mean(p_bid * bid_d + p_ask * ask_d - 0.5 * gamma * sigma**2 * q**2 * dt)
                objective += exp_reward
                
                # Update inventory
                # q_new = q + u_ask - u_bid
                q = q + u_ask.astype(float) - u_bid.astype(float)
                
                # Clip? The model says "no fill at q=Q". 
                # If q=Q, u_bid=0. So q can only decrease or stay same.
                # If q=Q-1, u_bid=1, q becomes Q.
                # So q stays in [-Q, Q].
                
            if objective > best_score:
                best_score = objective
                best_d0 = d0
                best_m = m

    # Fine Grid
    d0_fine = np.linspace(best_d0 - 0.2/k, best_d0 + 0.2/k, 10)
    m_fine = np.linspace(best_m - 0.005, best_m + 0.005, 10)
    
    N_fine = 500
    
    for d0 in d0_fine:
        for m in m_fine:
            q = np.zeros(N_fine)
            objective = 0.0
            
            for step in range(steps):
                bid_d = np.maximum(0, d0 + m * q)
                ask_d = np.maximum(0, d0 - m * q)
                
                mask_bid = q < Q
                mask_ask = q > -Q
                
                p_bid_raw = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                p_ask_raw = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                
                p_bid = np.where(mask_bid, p_bid_raw, 0.0)
                p_ask = np.where(mask_ask, p_ask_raw, 0.0)
                
                u_bid = np.random.rand(N_fine) < p_bid
                u_ask = np.random.rand(N_fine) < p_ask
                
                exp_reward = np.mean(p_bid * bid_d + p_ask * ask_d - 0.5 * gamma * sigma**2 * q**2 * dt)
                objective += exp_reward
                
                q = q + u_ask.astype(float) - u_bid.astype(float)
            
            if objective > best_score:
                best_score = objective
                best_d0 = d0
                best_m = m

    _OPT_CACHE[key] = (best_d0, best_m)
    return _OPT_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed optimal affine parameters via MC rollout.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimized parameters
    d0, m = _optimize_regime_mc(sigma, A, k, gamma, T, steps, Q)
    
    # Apply affine policy
    q = inventory.astype(float)
    
    bid_depths = d0 + m * q
    ask_depths = d0 - m * q
    
    # Ensure valid depths
    # Min depth 0, max depth 5/k (arbitrary large bound)
    bid_depths = np.clip(bid_depths, 0.0, 5.0 / k)
    ask_depths = np.clip(ask_depths, 0.0, 5.0 / k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
