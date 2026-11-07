"""Program for optimal market making with inventory risk using pre-computed HJB lookup table."""
# EVOLVE-BLOCK-START
import numpy as np

# Pre-compute optimal policy lookup table using coarse-grained HJB backward induction
# The state space is (step, inventory) -> (bid_depth, ask_depth)

# Parameters from the task
T_TOTAL = 1.0
STEPS = 1000
Q_MAX = 30
DT = T_TOTAL / STEPS

# Regimes to handle
REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

# For each regime, pre-compute a lookup table mapping (step, inventory) to (bid_depth, ask_depth)
# We use a reduced depth grid for computation efficiency

def _compute_optimal_policy(sigma, A, k, gamma):
    """
    Compute optimal bid/ask depths for all (step, inventory) pairs using backward induction.
    
    Returns:
        bid_table: array of shape (STEPS, 2*Q_MAX+1) with optimal bid depths
        ask_table: array of shape (STEPS, 2*Q_MAX+1) with optimal ask depths
    """
    num_inv = 2 * Q_MAX + 1
    inv_range = np.arange(-Q_MAX, Q_MAX + 1)
    
    # Value function: V[step, inv_index] = optimal expected reward from step onwards
    # V[STEPS, :] = 0 (no terminal charge)
    V = np.zeros((STEPS + 1, num_inv))
    
    # Store optimal actions
    bid_table = np.zeros((STEPS, num_inv))
    ask_table = np.zeros((STEPS, num_inv))
    
    # Coarse depth grid for search
    # Depths should be in (0, 3/k) to be valid
    max_depth = 3.0 / k
    # Use a logarithmic-ish grid with more points near small depths
    num_depths = 20
    depth_grid = np.linspace(0.01, max_depth * 0.99, num_depths)
    
    # Backward induction
    for n in range(STEPS - 1, -1, -1):
        # Inventory cost for this step
        inv_cost = (gamma / 2.0) * sigma**2 * inv_range**2 * DT
        
        for inv_idx in range(num_inv):
            q = inv_range[inv_idx]
            best_val = -np.inf
            best_bid = 0.0
            best_ask = 0.0
            
            for bid_d in depth_grid:
                # Bid fill probability
                p_bid_fill = 1.0 - np.exp(-A * np.exp(-k * bid_d) * DT)
                
                for ask_d in depth_grid:
                    # Ask fill probability
                    p_ask_fill = 1.0 - np.exp(-A * np.exp(-k * ask_d) * DT)
                    
                    # Expected immediate reward
                    # Bid fill: earn bid_d, inventory increases by 1 (if q < Q)
                    # Ask fill: earn ask_d, inventory decreases by 1 (if q > -Q)
                    # Note: inventory in the model - need to check sign convention
                    # "A fill earns its depth and moves inventory one unit"
                    # Bid hit: inventory increases (buying), so q -> q+1
                    # Ask hit: inventory decreases (selling), so q -> q-1
                    
                    # No bid fill at q = Q, no ask fill at q = -Q
                    if q >= Q_MAX:
                        p_bid_fill = 0.0
                    if q <= -Q_MAX:
                        p_ask_fill = 0.0
                    
                    # Immediate reward from fills
                    imm_reward = p_bid_fill * bid_d + p_ask_fill * ask_d
                    
                    # Next inventory states
                    # Bid fill: q -> q+1
                    # Ask fill: q -> q-1
                    # Both: q -> q (bid and ask fills cancel)
                    # Neither: q -> q
                    
                    # Probabilities of each joint event
                    p_both = p_bid_fill * p_ask_fill
                    p_bid_only = p_bid_fill * (1.0 - p_ask_fill)
                    p_ask_only = (1.0 - p_bid_fill) * p_ask_fill
                    p_neither = (1.0 - p_bid_fill) * (1.0 - p_ask_fill)
                    
                    # Next inventory for each case
                    # Need to map to index
                    def next_idx(dq):
                        new_q = q + dq
                        if new_q < -Q_MAX or new_q > Q_MAX:
                            return -1  # invalid
                        return new_q + Q_MAX
                    
                    idx_bid = next_idx(1)
                    idx_ask = next_idx(-1)
                    idx_both = next_idx(0)
                    idx_neither = next_idx(0)
                    
                    expected_next_val = 0.0
                    valid = True
                    
                    if idx_bid >= 0:
                        expected_next_val += p_bid_only * V[n + 1, idx_bid]
                    else:
                        valid = False
                    if idx_ask >= 0:
                        expected_next_val += p_ask_only * V[n + 1, idx_ask]
                    else:
                        valid = False
                    if idx_both >= 0:
                        expected_next_val += p_both * V[n + 1, idx_both]
                    else:
                        valid = False
                    if idx_neither >= 0:
                        expected_next_val += p_neither * V[n + 1, idx_neither]
                    else:
                        valid = False
                    
                    if not valid:
                        continue
                    
                    total_val = imm_reward - inv_cost + expected_next_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_bid = bid_d
                        best_ask = ask_d
            
            V[n, inv_idx] = best_val
            bid_table[n, inv_idx] = best_bid
            ask_table[n, inv_idx] = best_ask
    
    return bid_table, ask_table


# Pre-compute for all regimes at import time
_policy_cache = {}

def _get_policy(sigma, A, k, gamma):
    key = (sigma, A, k, gamma)
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_policy(sigma, A, k, gamma)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with keys 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two float arrays of shape (4000,)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    # Get pre-computed policy
    bid_table, ask_table = _get_policy(sigma, A, k, gamma)
    
    # Determine current step
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = np.clip(step, 0, STEPS - 1)
    
    # Map inventory to index
    inv_indices = inventory.astype(int) + Q_MAX
    inv_indices = np.clip(inv_indices, 0, 2 * Q_MAX)
    
    # Look up depths
    bid_depths = bid_table[step, inv_indices]
    ask_depths = ask_table[step, inv_indices]
    
    # Ensure depths are valid (positive, finite)
    bid_depths = np.maximum(bid_depths, 0.01)
    ask_depths = np.maximum(ask_depths, 0.01)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
