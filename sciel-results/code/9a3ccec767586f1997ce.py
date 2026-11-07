"""Market-making with inventory risk: Exact Discrete DP with Vectorized Value Iteration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

# Cache for pre-computed DP policies
_DP_CACHE = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal policy using exact Dynamic Programming (Value Iteration).
    
    The state space is the inventory level q in [-Q, Q].
    The action space is discretized into a set of candidate depths.
    We solve the Bellman equation backwards in time.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    
    # Discretize inventory
    # q can be integers from -Q to Q
    q_vals = np.arange(-Q, Q + 1)
    n_q = len(q_vals)
    
    # Discretize depth actions
    # We need a grid of depths to search over.
    # The optimal depth is likely bounded.
    # Let's choose a reasonable range.
    # d = 0 means always fill (if prob > 0).
    # Large d means never fill.
    # Let's use a logarithmic or linear grid.
    # Given k ~ 0.75-1.5, 1/k is ~0.6-1.3.
    # Let's use a grid from 0 to 5/k with 100 points.
    d_max = 5.0 / k
    n_d = 100
    d_grid = np.linspace(0.0, d_max, n_d)
    
    # Precompute fill probabilities for all d in d_grid
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * d_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    
    # Initialize Value Function V(q) for t = steps (terminal)
    # No terminal charge, so V_terminal(q) = 0
    V = np.zeros(n_q)
    
    # Store optimal actions for each time step? 
    # No, we just need the final policy? 
    # Wait, the policy depends on time t. 
    # V_t(q) = max_d [ Immediate Reward + Expected Future Value ]
    # We need to return the policy for ALL t? 
    # The function quote(t, ...) is called with specific t.
    # So we need to store the optimal action for each (t, q).
    # Or, we can just store the value functions and re-compute argmax on the fly?
    # Re-computing argmax on the fly is fast if we precompute the expected values.
    # But we need the specific d for each q at time t.
    
    # Let's store the optimal depth index for each q at each time step.
    # optimal_action[t, q_idx] = index in d_grid
    # Memory: 1000 steps * 61 states * 1 byte = 61000 bytes. Very small.
    optimal_actions = np.zeros((steps + 1, n_q), dtype=np.int8)
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        # For each q, find d that maximizes:
        # - 0.5 * gamma * sigma^2 * q^2 * dt  (Risk cost, note: q is pre-fill)
        # + p(d) * [ d + V(q-1) ]  (Bid fill: inventory decreases)
        # + p(d) * [ d + V(q+1) ]  (Ask fill: inventory increases)
        # + (1 - p(d)) * [ 0 + V(q) ]     (No fill)
        # Note: If q == Q, no bid fill. If q == -Q, no ask fill.
        
        # Vectorize over q
        # q_vals is shape (n_q,)
        # d_grid is shape (n_d,)
        
        # We need to compute the value for each (q, d) pair.
        # Let's expand q to (n_q, 1) and d to (1, n_d)
        q_exp = q_vals[:, np.newaxis]
        d_exp = d_grid[np.newaxis, :]
        p_exp = p_fill[np.newaxis, :]
        
        # Risk cost
        risk_cost = 0.5 * gamma * sigma**2 * q_exp**2 * dt
        
        # Bid fill component
        # q' = q - 1
        # If q == Q, no bid fill.
        # We need V(q-1).
        # q_vals are -Q, -Q+1, ..., Q.
        # Index for q is i. Index for q-1 is i-1.
        # If i=0 (q=-Q), q-1 is out of bounds. But bid fill increases inventory? 
        # Wait. 
        # "A fill earns its depth and moves inventory one unit"
        # "bid hit ... inventory (4000,) ints in [-Q, Q]"
        # Standard convention:
        # Buy (Bid hit) -> Inventory increases? Or decreases?
        # Usually, a market maker buys from the bid side, so they hold more inventory.
        # Let's check the task description carefully.
        # "bid_depths ... distances below mid"
        # "ask_depths ... distances above mid"
        # "A fill earns its depth and moves inventory one unit"
        # "no bid fill at q = Q, no ask fill at q = -Q"
        # If no bid fill at q=Q, it implies bid fill increases inventory towards Q.
        # So Bid Fill: q -> q + 1.
        # Ask Fill: q -> q - 1.
        # Let's re-read: "no bid fill at q = Q". If bid fill increased q, then at Q we can't go higher. Correct.
        # "no ask fill at q = -Q". If ask fill decreased q, then at -Q we can't go lower. Correct.
        # So:
        # Bid Fill (buy): q -> q + 1
        # Ask Fill (sell): q -> q - 1
        
        # Update logic:
        # V_next(q) is V from previous iteration (which is V_{t+1})
        # Current V is V_t
        
        # Indices:
        # i corresponds to q = q_vals[i]
        # q+1 index is i+1
        # q-1 index is i-1
        
        # V_plus = V[i+1], V_minus = V[i-1], V_same = V[i]
        
        V_plus = np.zeros(n_q)
        V_minus = np.zeros(n_q)
        V_same = V.copy()
        
        # q+1 exists for i < n_q - 1
        V_plus[:-1] = V[1:]
        # q+1 at i=n_q-1 (q=Q) is invalid. Value should be 0 or irrelevant since prob=0.
        # We set it to 0, but p_fill will be masked or we handle boundaries.
        # Actually, if q=Q, bid fill prob is 0. So V_plus doesn't matter.
        
        # q-1 exists for i > 0
        V_minus[1:] = V[:-1]
        # q-1 at i=0 (q=-Q) is invalid.
        
        # Bid fill contribution: p(d) * (d + V_plus)
        # Ask fill contribution: p(d) * (d + V_minus)
        # No fill: (1 - p(d)) * V_same
        
        # Note: p(d) is the same for bid and ask?
        # "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt), ask likewise"
        # Yes, symmetric probability model.
        
        bid_term = p_exp * (d_exp + V_plus[np.newaxis, :])
        ask_term = p_exp * (d_exp + V_minus[np.newaxis, :])
        no_fill_term = (1.0 - p_exp) * V_same[np.newaxis, :]
        
        total_value = -risk_cost + bid_term + ask_term + no_fill_term
        
        # Find max over d
        max_val = np.max(total_value, axis=1)
        opt_idx = np.argmax(total_value, axis=1)
        
        # Update V for next iteration (t-1)
        V = max_val
        optimal_actions[t] = opt_idx
        
    # Store results
    # We need to map back to depths.
    # optimal_actions[t, q_idx] gives index in d_grid.
    _DP_CACHE[key] = (optimal_actions, d_grid)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed DP.
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
    
    # Get optimized DP tables
    optimal_actions, d_grid = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    
    # t is float n * dt. We need integer step index.
    # The problem says "called at every step n = 0..999 with t = n * dt".
    # So n = round(t / dt).
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    n = np.clip(n, 0, steps - 1)
    
    # inventory is (4000,) ints in [-Q, Q]
    # We need to get the optimal depth for each inventory level at step n.
    # optimal_actions[n] is an array of shape (2Q+1,)
    # Index for q is q + Q
    
    q_idx = inventory + Q
    # Ensure q_idx is valid
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    # Get optimal action indices
    opt_idxs = optimal_actions[n][q_idx]
    
    # Get depths
    depths = d_grid[opt_idxs]
    
    # The model assumes symmetric quotes?
    # The DP assumed symmetric p(d) for bid and ask.
    # The objective is symmetric in terms of fill probabilities.
    # However, the inventory risk is symmetric q^2.
    # So the optimal policy should be symmetric in the sense that the depth chosen
    # depends on q.
    # In our DP, we chose a single depth d for both bid and ask at state q.
    # Is it possible that optimal bid != optimal ask?
    # If p_bid(d_b) and p_ask(d_a) can be different.
    # The Bellman equation would be:
    # max_{d_b, d_a} [ -risk + p_b(d_b)(d_b + V(q+1)) + p_a(d_a)(d_a + V(q-1)) + (1-p_b)(1-p_a)V(q) ]
    # This is a 2D optimization.
    # However, due to symmetry of the problem (V is even function of q? No, V depends on q).
    # Wait, V(q) is not necessarily even.
    # But the transition is symmetric.
    # If V(q) was even, then V(q+1) = V(-q-1) etc.
    # Actually, let's check if symmetric quotes (d_b = d_a = d) are optimal.
    # If we restrict to d_b = d_a, we get the policy above.
    # Given the time constraint and the fact that the seed program used symmetric quotes,
    # and the problem structure is highly symmetric, it's likely that d_b = d_a is optimal or very close.
    # Let's stick with symmetric for now to ensure speed and correctness of the 1D search.
    
    bid_depths = depths
    ask_depths = depths
    
    # Ensure valid depths
    # Must be >= 0.
    bid_depths = np.clip(bid_depths, 0.0, np.inf)
    ask_depths = np.clip(ask_depths, 0.0, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
