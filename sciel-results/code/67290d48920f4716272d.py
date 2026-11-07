"""Market-making with inventory risk: Analytical LQ Control Approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

# Cache for optimized parameters per regime
_DP_CACHE = {}

def _solve_lq(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal policy using Linear-Quadratic (LQ) control theory.
    
    We assume the value function is quadratic in inventory: V(q) = c0 + c1 q + c2 q^2.
    Due to symmetry, c1 should be 0. So V(q) = c0 + c2 q^2.
    
    The optimal policy for LQ systems is affine: d(q) = d0 + d1 q.
    
    We derive the coefficients d0, d1 and c2 by solving the Bellman equation
    in the limit where the dynamics are linearized around the optimal policy.
    
    Actually, for this specific non-linear fill probability, we can use a
    first-order approximation or solve for the fixed point of the coefficients.
    
    Let's use the "Average Reward" or "Infinite Horizon" approximation which is
    often very accurate for stationary regimes, or simply solve the DARE for the
    discrete time step dt.
    
    However, the fill probability is non-linear in depth.
    p(d) = 1 - exp(-A exp(-k d) dt).
    
    Let's use a simpler, robust heuristic derived from LQ intuition:
    1. The optimal depth has a base component (risk-neutral) and a skew component (inventory risk).
    2. d(q) = d_base + d_skew * q.
    
    We can find d_base and d_skew by minimizing the expected cost over a short horizon
    or by matching the first-order conditions of the continuous-time LQ problem,
    adjusted for the discrete fill dynamics.
    
    Continuous-time LQ Market Making (Avellaneda-Stoikov style):
    Optimal spread is related to risk aversion and volatility.
    Here, we have explicit fill probabilities.
    
    Let's perform a fast local optimization for d0 and d1 for each regime.
    We approximate the long-run average cost by simulating a short horizon (e.g., 100 steps)
    with a small number of paths (e.g., 100 paths) to evaluate the cost of a given (d0, d1).
    Then we use Nelder-Mead or simple grid search to find the best (d0, d1).
    This is much faster than full DP because we only optimize 2 parameters.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    
    # We will search for d0 (base depth) and d1 (skew).
    # d(q) = d0 + d1 * q
    # Constraints: d(q) >= 0 for q in [-Q, Q].
    # So d0 >= |d1| * Q.
    
    # Let's do a coarse grid search over d0 and d1, then refine.
    # Range for d0: [0, 5/k]
    # Range for d1: [-1, 1] (scaled by 1/k? No, let's keep it raw)
    
    # Actually, let's use a more direct approach: 
    # The optimal depth for a given inventory q can be approximated by balancing
    # the expected gain from fills against the risk cost.
    
    # Let's try a small set of candidate affine policies and evaluate their performance
    # using a Monte Carlo simulation of a short horizon (e.g., 50 steps) to estimate
    # the average reward.
    
    # Candidates for d0: [0.5/k, 1.0/k, 1.5/k, 2.0/k, 2.5/k]
    # Candidates for d1: [-0.2, -0.1, 0, 0.1, 0.2] (times some scale factor?)
    # The skew should be proportional to gamma * sigma^2.
    
    # Let's define a function to evaluate a policy (d0, d1)
    def evaluate_policy(d0, d1, n_paths=100, n_steps=100):
        # Simulate n_paths, n_steps
        # Inventory starts at 0 for all paths? Or random?
        # To be fair, start at 0.
        inventories = np.zeros(n_paths)
        total_reward = 0.0
        
        for _ in range(n_steps):
            # Generate quotes
            depths = d0 + d1 * inventories
            depths = np.clip(depths, 0.0, np.inf)
            
            # Calculate fill probabilities
            # p = 1 - exp(-A * exp(-k * depth) * dt)
            # Handle depth=0 carefully? exp(0)=1, so p = 1-exp(-A*dt).
            # If depth is very large, exp(-k*depth) is 0, p=0.
            
            # Clip depths to avoid overflow in exp
            depths_clipped = np.clip(depths, 0.0, 10.0/k)
            
            exp_term = np.exp(-k * depths_clipped)
            p_fill = 1.0 - np.exp(-A * exp_term * dt)
            
            # Determine fills
            # Bid fill: inventory decreases by 1. Reward: depth.
            # Ask fill: inventory increases by 1. Reward: depth.
            # We assume symmetric quotes for bid and ask for simplicity in this eval?
            # The task asks for bid_depths and ask_depths. 
            # In LQ, often bid = ask = d(q). 
            # Let's assume symmetric quotes for the evaluation to keep it simple.
            # If we want asymmetric, we'd need 4 params. 2 params is a good start.
            
            # Random draws
            u1 = np.random.rand(n_paths)
            u2 = np.random.rand(n_paths)
            
            bid_fill = (u1 < p_fill) & (inventories < Q)
            ask_fill = (u2 < p_fill) & (inventories > -Q)
            
            # Update inventory
            inventories = inventories - bid_fill.astype(float) + ask_fill.astype(float)
            
            # Reward
            # Depth earned
            reward = (bid_fill * depths) + (ask_fill * depths)
            # Risk cost
            risk_cost = 0.5 * gamma * sigma**2 * inventories**2 * dt
            # Note: risk cost is applied to inventory *before* or *after*?
            # "sum over steps of (gamma / 2) sigma^2 q^2 dt, q the inventory before that step's fills"
            # So we should use the inventory *before* update.
            # Let's correct the loop order.
            
            # Re-do reward calculation with pre-fill inventory
            # We already updated inventories above. Let's store pre-fill.
            # Actually, the loop above is slightly wrong for the risk cost timing.
            # Let's fix the evaluation function.
            
        return total_reward / n_paths # This is a dummy return, need to fix logic

    # Let's rewrite the evaluation function properly.
    def evaluate_policy_v2(d0, d1, n_paths=200, n_steps=50):
        inventories = np.zeros(n_paths)
        total_reward = 0.0
        
        for _ in range(n_steps):
            pre_inv = inventories.copy()
            
            depths = d0 + d1 * pre_inv
            depths = np.clip(depths, 0.0, 10.0/k)
            
            exp_term = np.exp(-k * depths)
            p_fill = 1.0 - np.exp(-A * exp_term * dt)
            
            u1 = np.random.rand(n_paths)
            u2 = np.random.rand(n_paths)
            
            bid_fill = (u1 < p_fill) & (pre_inv < Q)
            ask_fill = (u2 < p_fill) & (pre_inv > -Q)
            
            # Reward
            reward = (bid_fill * depths) + (ask_fill * depths)
            risk_cost = 0.5 * gamma * sigma**2 * pre_inv**2 * dt
            
            total_reward += np.sum(reward - risk_cost)
            
            # Update inventory
            inventories = pre_inv - bid_fill.astype(float) + ask_fill.astype(float)
            
        return total_reward / (n_paths * n_steps)

    # Grid search for d0, d1
    # d0 range: [0.1/k, 3.0/k]
    # d1 range: [-0.5, 0.5] * (gamma * sigma**2) ? No, d1 should be small.
    # Let's guess d1 is proportional to gamma * sigma^2 * dt / (something).
    # For now, search d1 in [-0.2, 0.2].
    
    d0_grid = np.linspace(0.5/k, 2.5/k, 10)
    d1_grid = np.linspace(-0.1, 0.1, 10)
    
    best_score = -np.inf
    best_d0 = 1.0/k
    best_d1 = 0.0
    
    # Set seed for reproducibility in evaluation
    np.random.seed(42)
    
    for d0 in d0_grid:
        for d1 in d1_grid:
            score = evaluate_policy_v2(d0, d1)
            if score > best_score:
                best_score = score
                best_d0 = d0
                best_d1 = d1
    
    # Refine around best
    d0_fine = np.linspace(best_d0 - 0.2/k, best_d0 + 0.2/k, 5)
    d1_fine = np.linspace(best_d1 - 0.05, best_d1 + 0.05, 5)
    
    for d0 in d0_fine:
        for d1 in d1_fine:
            score = evaluate_policy_v2(d0, d1)
            if score > best_score:
                best_score = score
                best_d0 = d0
                best_d1 = d1
    
    # Store the affine policy parameters
    _DP_CACHE[key] = (best_d0, best_d1)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed LQ affine parameters.
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
    
    # Get optimized affine parameters
    d0, d1 = _solve_lq(sigma, A, k, gamma, T, steps, Q)
    
    # Calculate depths for current inventory
    # d(q) = d0 + d1 * q
    bid_depths = d0 + d1 * inventory
    ask_depths = d0 + d1 * inventory # Symmetric quotes for now
    
    # Ensure valid depths
    # Must be >= 0.
    bid_depths = np.clip(bid_depths, 0.0, np.inf)
    ask_depths = np.clip(ask_depths, 0.0, np.inf)
    
    # Ensure shape is (4000,)
    # inventory is (4000,), so depths are (4000,).
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
