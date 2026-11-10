"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for pre-computed policies
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIMEOUT_LIMIT = 320.0  # Leave 10s margin for 330s limit

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths for each (time, inventory) state using DP.
    Uses decoupled approximation for speed: ignores p_b * p_a term in expectation.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Check time
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        # Fallback: return None to signal timeout, quote will use heuristic
        return None

    dt = T / steps
    num_q = 2 * Q + 1
    # V_next[q_index] where q_index = q + Q
    V_next = np.zeros(num_q)
    
    # Pre-compute depth grid
    # Optimal depth is around 1/k. Range [0, 2.0] should cover it.
    # Use 100 points for good resolution.
    depth_grid = np.linspace(0.0, 2.0, 100)
    
    # Policy storage: policy[n, q_index] = (bid_depth, ask_depth)
    # We only need the final policy for n=0..steps-1
    # But we compute backwards, so we store all steps in a list or array
    policy = np.zeros((steps, num_q, 2))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_q)
        for qi in range(num_q):
            q = qi - Q
            V0 = V_next[qi]
            
            # Determine neighbors
            # If q == Q, no bid fill possible. q-1 is Q-1, valid.
            # If q == -Q, no ask fill possible. q+1 is -Q+1, valid.
            
            bid_opt = 0.0
            ask_opt = 0.0
            
            # Optimize Bid Depth
            if q < Q:
                # p_b = 1 - exp(-A * exp(-k * b) * dt)
                # Objective for b: b * p_b + (1 - p_b) * V0 + p_b * V_next[qi - 1]
                # Note: V_next[qi - 1] corresponds to q - 1
                V_minus = V_next[qi - 1]
                best_val = -np.inf
                for b in depth_grid:
                    if b < 0: continue
                    exp_term = np.exp(-k * b)
                    p_b = 1.0 - np.exp(-A * exp_term * dt)
                    # Decoupled expectation term for bid:
                    # (1-p_b)*V0 + p_b*V_minus
                    # Reward: b * p_b
                    val = b * p_b + (1.0 - p_b) * V0 + p_b * V_minus
                    if val > best_val:
                        best_val = val
                        bid_opt = b
            
            # Optimize Ask Depth
            if q > -Q:
                # p_a = 1 - exp(-A * exp(-k * a) * dt)
                # Objective for a: a * p_a + (1 - p_a) * V0 + p_a * V_next[qi + 1]
                V_plus = V_next[qi + 1]
                best_val = -np.inf
                for a in depth_grid:
                    if a < 0: continue
                    exp_term = np.exp(-k * a)
                    p_a = 1.0 - np.exp(-A * exp_term * dt)
                    val = a * p_a + (1.0 - p_a) * V0 + p_a * V_plus
                    if val > best_val:
                        best_val = val
                        ask_opt = a
            
            # Calculate current value V_curr[qi]
            # Using the optimal b and a found
            if q < Q and q > -Q:
                exp_b = np.exp(-k * bid_opt)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                exp_a = np.exp(-k * ask_opt)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                
                V_minus = V_next[qi - 1]
                V_plus = V_next[qi + 1]
                
                # Full expectation including coupling term p_b * p_a * V0
                # E[V] = (1-p_b)(1-p_a)V0 + p_b(1-p_a)V_minus + (1-p_b)p_a V_plus + p_b p_a V0
                # = V0 (1 - p_b - p_a + 2 p_b p_a) + p_b V_minus (1 - p_a) + p_a V_plus (1 - p_b)
                # Wait, the decoupled approximation ignored the -p_b p_a V0 and the (1-p_a) on V_minus etc.
                # For consistency in value calculation, we should use the full model.
                # But for optimization we used decoupled. This is fine.
                
                E_V = ((1.0 - p_b) * (1.0 - p_a) + p_b * p_a) * V0 + \
                      p_b * (1.0 - p_a) * V_minus + \
                      (1.0 - p_b) * p_a * V_plus
                
                reward = bid_opt * p_b + ask_opt * p_a - 0.5 * gamma * sigma**2 * q**2 * dt
                V_curr[qi] = reward + E_V
                
            elif q == Q:
                # Only ask side
                if ask_opt > 0:
                    exp_a = np.exp(-k * ask_opt)
                    p_a = 1.0 - np.exp(-A * exp_a * dt)
                    V_plus = V_next[qi + 1]
                    E_V = (1.0 - p_a) * V0 + p_a * V_plus
                    reward = ask_opt * p_a - 0.5 * gamma * sigma**2 * q**2 * dt
                    V_curr[qi] = reward + E_V
                else:
                    V_curr[qi] = -0.5 * gamma * sigma**2 * q**2 * dt
                    # No fills, state stays same? No, if no fill, state is same.
                    # But we said no ask fill at -Q. At Q, no bid fill.
                    # If ask_opt is 0, p_a might be small but non-zero?
                    # In our grid, if optimal is 0, we set ask_opt=0.
                    # If we set depth 0, p = 1 - exp(-A * 1 * dt) > 0.
                    # So we must evaluate depth 0 properly.
                    # The loop above handles b=0.
                    # So ask_opt will be the best from grid, which includes 0.
                    # So the 'if ask_opt > 0' check is wrong. We should always compute.
                    pass # Handled in general case if we don't skip.
            
            elif q == -Q:
                # Only bid side
                if bid_opt > 0:
                    exp_b = np.exp(-k * bid_opt)
                    p_b = 1.0 - np.exp(-A * exp_b * dt)
                    V_minus = V_next[qi - 1]
                    E_V = (1.0 - p_b) * V0 + p_b * V_minus
                    reward = bid_opt * p_b - 0.5 * gamma * sigma**2 * q**2 * dt
                    V_curr[qi] = reward + E_V
                else:
                    V_curr[qi] = -0.5 * gamma * sigma**2 * q**2 * dt

            # Correction: The logic above for boundaries was slightly messy.
            # Let's re-structure the value calculation to be robust.
            
            # Recalculate V_curr[qi] robustly
            V_curr_val = 0.0
            if q < Q and q > -Q:
                exp_b = np.exp(-k * bid_opt)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                exp_a = np.exp(-k * ask_opt)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                V_minus = V_next[qi - 1]
                V_plus = V_next[qi + 1]
                E_V = ((1.0 - p_b) * (1.0 - p_a) + p_b * p_a) * V0 + \
                      p_b * (1.0 - p_a) * V_minus + \
                      (1.0 - p_b) * p_a * V_plus
                reward = bid_opt * p_b + ask_opt * p_a - 0.5 * gamma * sigma**2 * q**2 * dt
                V_curr_val = reward + E_V
            elif q == Q:
                exp_a = np.exp(-k * ask_opt)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                V_plus = V_next[qi + 1]
                E_V = (1.0 - p_a) * V0 + p_a * V_plus
                reward = ask_opt * p_a - 0.5 * gamma * sigma**2 * q**2 * dt
                V_curr_val = reward + E_V
            elif q == -Q:
                exp_b = np.exp(-k * bid_opt)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                V_minus = V_next[qi - 1]
                E_V = (1.0 - p_b) * V0 + p_b * V_minus
                reward = bid_opt * p_b - 0.5 * gamma * sigma**2 * q**2 * dt
                V_curr_val = reward + E_V
            
            V_curr[qi] = V_curr_val
            policy[n, qi, 0] = bid_opt
            policy[n, qi, 1] = ask_opt
            
        V_next = V_curr
        
    _POLICY_CACHE[key] = policy
    return policy


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths for each path.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    policy = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
    
    bid_depths = np.full(len(inventory), np.inf)
    ask_depths = np.full(len(inventory), np.inf)
    
    if policy is None:
        # Fallback heuristic if DP timed out
        # Use simple skew based on inventory
        for i, q in enumerate(inventory):
            # If q > 0, we are long, so we want to sell more (wider ask, tighter bid? No, tighter ask to sell)
            # Actually, if long, we want to reduce inventory. So make ask more attractive (smaller depth) and bid less attractive (larger depth).
            # Depth is distance from mid. Smaller depth = closer to mid = more likely to fill.
            # So for long (q>0): Ask depth small, Bid depth large.
            # For short (q<0): Bid depth small, Ask depth large.
            base_depth = 1.0 / k
            skew = 0.1 * q / Q
            bid_d = base_depth * (1.0 + skew)
            ask_d = base_depth * (1.0 - skew)
            # Ensure non-negative
            if bid_d < 0: bid_d = 0.0
            if ask_d < 0: ask_d = 0.0
            
            if q < Q:
                bid_depths[i] = bid_d
            if q > -Q:
                ask_depths[i] = ask_d
        return bid_depths, ask_depths

    # Extract depths from policy
    for i, q in enumerate(inventory):
        qi = q + Q
        if 0 <= qi < 2 * Q + 1:
            b = policy[n, qi, 0]
            a = policy[n, qi, 1]
            
            if q < Q:
                bid_depths[i] = b
            if q > -Q:
                ask_depths[i] = a
        else:
            # Should not happen
            bid_depths[i] = 0.0
            ask_depths[i] = 0.0
            
    # Ensure no negative depths
    bid_depths[bid_depths < 0] = 0.0
    ask_depths[ask_depths < 0] = 0.0
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
