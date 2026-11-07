# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0
_POLICY_CACHE = {}


def _compute_policy(params):
    """
    Computes the optimal bid and ask depths for each inventory level and time step
    using backward induction (discrete HJB).
    """
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps
    
    num_q = 2 * Q + 1
    
    # V_next[i] is the optimal expected future reward from step n+1 onwards
    # starting with inventory q = i - Q.
    V_next = np.zeros(num_q)
    
    # Tables of optimal depths
    bid_depths = np.zeros((steps, num_q))
    ask_depths = np.zeros((steps, num_q))
    
    # Pre-compute the optimal depth for a given "incentive" C.
    # The sub-problem for bid is: max_x [ (1 - exp(-A exp(-k x))) * (x + C) ]
    # where C = DeltaV.
    # We create a lookup table for C in a reasonable range.
    
    # Range of C:
    # C = V(q+1) - V(q).
    # V is concave, penalty ~ -gamma/2 sigma^2 q^2.
    # DeltaV ~ -gamma sigma^2 q.
    # Max |q| = 30. gamma max 0.5, sigma max 2.
    # |C| ~ 0.5 * 4 * 30 = 60.
    # Let's use range [-100, 100] to be safe.
    c_min = -100.0
    c_max = 100.0
    num_c = 200
    c_grid = np.linspace(c_min, c_max, num_c)
    x_grid = np.zeros(num_c)
    
    # Optimization bounds
    # x in [0, x_max].
    # At x_max, p(x) ~ 0.
    # A exp(-k x_max) = 1e-6 => x_max = (1/k) ln(A/1e-6).
    # For A=140, k=1.5: x_max ~ 12.
    # For A=50, k=0.75: x_max ~ 13.
    x_max = (1.0 / k) * np.log(A / 1e-6) + 1.0  # +1 buffer
    
    def objective(x, C):
        p = 1.0 - np.exp(-A * np.exp(-k * x))
        return p * (x + C)
    
    # Compute optimal x for each C
    for idx, C in enumerate(c_grid):
        # If C is very negative, optimum is large x (low prob).
        # If C is very positive, optimum is small x (high prob).
        # Use bounded minimization of -objective.
        res = minimize_scalar(lambda x: -objective(x, C), bounds=(0.0, x_max), method='bounded')
        x_grid[idx] = res.x
        
    # Ensure monotonicity? C increasing -> x decreasing.
    # If not, interpolate carefully or fix.
    # Let's just use interp.
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_current = np.zeros(num_q)
        # We process all inventory levels.
        # Note: The decision at step n depends on V_next.
        
        for i in range(num_q):
            q = i - Q
            
            # Determine optimal bid depth
            if q == Q:
                x_bid = np.inf
                p_bid = 0.0
            else:
                # C_bid = V_next(q+1) - V_next(q)
                C_bid = V_next[i + 1] - V_next[i]
                # Clamp C to grid range
                C_bid_clipped = np.clip(C_bid, c_min, c_max)
                x_bid = np.interp(C_bid_clipped, c_grid, x_grid)
                p_bid = 1.0 - np.exp(-A * np.exp(-k * x_bid))
            
            # Determine optimal ask depth
            if q == -Q:
                x_ask = np.inf
                p_ask = 0.0
            else:
                # C_ask = V_next(q-1) - V_next(q)
                C_ask = V_next[i - 1] - V_next[i]
                C_ask_clipped = np.clip(C_ask, c_min, c_max)
                x_ask = np.interp(C_ask_clipped, c_grid, x_grid)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * x_ask))
            
            # Store depths
            bid_depths[n, i] = x_bid
            ask_depths[n, i] = x_ask
            
            # Calculate V_current
            # Transition probabilities:
            # p_b = p_bid, p_a = p_ask
            # P(q+1) = p_b (1 - p_a)  (Bid hit, Ask miss)
            # P(q-1) = (1 - p_b) p_a  (Bid miss, Ask hit)
            # P(q)   = 1 - P(q+1) - P(q-1) = (1 - p_b)(1 - p_a) + p_b p_a
            
            # Reward:
            # If bid hit: earn x_bid.
            # If ask hit: earn x_ask.
            # Inventory penalty: -gamma/2 sigma^2 q^2 dt.
            
            V_val = 0.0
            if q < Q:
                V_val += p_bid * (1 - p_ask) * (x_bid + V_next[i + 1])
            else:
                # At q=Q, no bid fill. But p_bid is 0 anyway.
                # So term is 0.
                pass
                
            if q > -Q:
                V_val += (1 - p_bid) * p_ask * (x_ask + V_next[i - 1])
            else:
                # At q=-Q, no ask fill. p_ask is 0.
                pass
                
            # Stay at q
            p_stay = 1.0 - (p_bid * (1 - p_ask) + (1 - p_bid) * p_ask)
            # p_stay = (1-p_b)(1-p_a) + p_b p_a
            V_val += p_stay * V_next[i]
            
            # Add immediate rewards from fills?
            # Wait, the above terms x_bid + V_next[i+1] assume that if we transition,
            # we get the reward x_bid AND the value V_next.
            # Is this correct?
            # Yes, the reward is realized at the transition.
            
            # Add inventory penalty
            V_val -= 0.5 * gamma * sigma**2 * q**2 * dt
            
            V_current[i] = V_val
        
        V_next = V_current
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Returns bid and ask depths for the given state.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe default: moderate spread
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy(params)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    steps = params['steps']
    Q = params['Q']
    T = params['T']
    dt = T / steps
    
    # Determine current step
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    indices = inventory + Q
    
    # Retrieve depths
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
