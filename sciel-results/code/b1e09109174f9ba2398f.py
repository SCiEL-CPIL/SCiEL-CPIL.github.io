"""Optimal market making with inventory risk - analytic backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 330s kill

_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction with analytic depth optimization.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    n_states = 2 * Q + 1
    
    # Initialize V at terminal step
    V = np.zeros((steps + 1, n_states))
    
    # Bid and ask depth tables
    bid_table = np.zeros((steps, n_states))
    ask_table = np.zeros((steps, n_states))

    # Precompute constants
    log_A_dt = np.log(A * dt)
    # p_max corresponds to d_min. 
    # Valid d > -3/k. Let's use d_min = 0.01 for safety, or slightly above -3/k.
    # d_min = -3/k is the limit. We should stay strictly above.
    # Let's use d_min = max(0.0, -3.0/k + 0.01)
    d_min = max(0.0, -3.0 / k + 0.01)
    p_max = 1.0 - np.exp(-A * np.exp(-k * d_min) * dt)
    
    # Helper to convert p to depth
    def p_to_depth(p):
        if p <= 0:
            return np.inf
        if p >= 1.0:
            return 0.0 # Theoretical limit, but p<1 always
        # p = 1 - exp(-A exp(-k d) dt)
        # 1-p = exp(-A exp(-k d) dt)
        # ln(1-p) = -A exp(-k d) dt
        # -ln(1-p) / (A dt) = exp(-k d)
        # ln(-ln(1-p) / (A dt)) = -k d
        # d = -1/k * ln(-ln(1-p) / (A dt))
        # d = 1/k * ln(A dt / -ln(1-p))
        val = np.log(A * dt / (-np.log(1.0 - p))) / k
        if val < d_min:
            return d_min
        return val

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            # Fallback: fill remaining with simple heuristic
            # Use last computed or default
            break
            
        V_next = V[n + 1]
        risk_cost = 0.5 * gamma * sigma * sigma * dt

        for qi in range(n_states):
            q = qi - Q
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                V[n, qi] = -risk_cost * q * q
                bid_table[n, qi] = np.inf
                ask_table[n, qi] = np.inf
                continue
            
            # Get deltas
            # V_next indices: q -> qi
            # q+1 -> qi+1 (if q < Q)
            # q-1 -> qi-1 (if q > -Q)
            
            if can_bid:
                delta_plus = V_next[qi + 1] - V_next[qi]
            else:
                delta_plus = 0.0 # Not used if can_bid is false, but keep safe
                
            if can_ask:
                delta_minus = V_next[qi - 1] - V_next[qi]
            else:
                delta_minus = 0.0

            K = delta_plus + delta_minus
            
            # Determine optimal p_b, p_a
            # C_b = b + delta_plus
            # C_a = a + delta_minus
            
            # We need to solve for p_b, p_a.
            # If K > 0, we have an interior solution candidate.
            # If K <= 0, we check boundaries.
            
            best_val = -np.inf
            best_pb = 0.0
            best_pa = 0.0
            
            # Define candidate (pb, pa) pairs
            candidates = []
            
            if can_bid and can_ask:
                if K > 1e-12:
                    # Interior solution
                    # Initialize p_b, p_a
                    pb = 0.1 * p_max
                    pa = 0.1 * p_max
                    
                    for _ in range(10):
                        # Compute C_b, C_a
                        b = p_to_depth(pb)
                        a = p_to_depth(pa)
                        C_b = b + delta_plus
                        C_a = a + delta_minus
                        
                        # Update
                        pb_new = C_a / K
                        pa_new = C_b / K
                        
                        # Clamp
                        pb_new = np.clip(pb_new, 0.0, p_max)
                        pa_new = np.clip(pa_new, 0.0, p_max)
                        
                        if abs(pb_new - pb) < 1e-8 and abs(pa_new - pa) < 1e-8:
                            break
                        pb = pb_new
                        pa = pa_new
                    
                    candidates.append((pb, pa))
                    
                    # Also check boundaries in case interior is not global max
                    # (e.g. if C_b or C_a becomes negative during iteration)
                    candidates.append((0.0, p_max))
                    candidates.append((p_max, 0.0))
                    candidates.append((0.0, 0.0))
                    candidates.append((p_max, p_max))
                    
                else:
                    # K <= 0, convex function, max at corners
                    candidates.append((0.0, 0.0))
                    candidates.append((p_max, 0.0))
                    candidates.append((0.0, p_max))
                    candidates.append((p_max, p_max))
            elif can_bid:
                # Only bid
                # Maximize p_b (b + delta_plus)
                # If b + delta_plus > 0, use max p_b, else 0
                # b depends on p_b.
                # f(p) = p (b(p) + delta_plus)
                # Derivative: b + delta_plus + p b'(p)
                # b'(p) = 1/k * d/dp ln(A dt / -ln(1-p)) = 1/k * (1 / (-ln(1-p))) * (1/(1-p)) * (-1) * (-1) ?
                # d/dp (-ln(1-p)) = 1/(1-p)
                # d/dp ln(1/L) = -1/L * L' = -1/L * 1/(1-p)
                # So b'(p) = 1/k * (-1 / (-ln(1-p)) * 1/(1-p)) = -1 / (k (1-p) (-ln(1-p)))
                # Wait, b(p) = 1/k ln(A dt / L(p)). L(p) = -ln(1-p).
                # b'(p) = 1/k * (-1/L(p)) * L'(p) = 1/k * (-1/L(p)) * 1/(1-p) = -1 / (k (1-p) L(p))
                # Since L(p) > 0, b'(p) < 0.
                # f'(p) = C_b(p) + p b'(p).
                # We can just check p=0 and p=p_max.
                candidates.append((0.0, 0.0))
                candidates.append((p_max, 0.0))
            elif can_ask:
                candidates.append((0.0, 0.0))
                candidates.append((0.0, p_max))
            
            # Evaluate candidates
            for pb, pa in candidates:
                if pb < -1e-9 or pb > p_max + 1e-9 or pa < -1e-9 or pa > p_max + 1e-9:
                    continue
                if pb < 0: pb = 0.0
                if pb > p_max: pb = p_max
                if pa < 0: pa = 0.0
                if pa > p_max: pa = p_max
                
                b = p_to_depth(pb)
                a = p_to_depth(pa)
                
                val = -risk_cost * q * q
                val += pb * b + pa * a
                val += (1 - pb) * (1 - pa) * V_next[qi]
                if can_bid:
                    val += pb * (1 - pa) * V_next[qi + 1]
                if can_ask:
                    val += pa * (1 - pb) * V_next[qi - 1]
                # Both fills
                if can_bid and can_ask:
                    val += pb * pa * V_next[qi]
                
                if val > best_val:
                    best_val = val
                    best_pb = pb
                    best_pa = pa
            
            V[n, qi] = best_val
            bid_table[n, qi] = p_to_depth(best_pb) if best_pb > 0 else np.inf
            ask_table[n, qi] = p_to_depth(best_pa) if best_pa > 0 else np.inf

    _OPT_TABLES[key] = (bid_table, ask_table)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths using pre-computed backward induction table.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple heuristic
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        steps = params['steps']
        T = params['T']
        dt = T / steps
        q = inventory.astype(np.float64)
        d_base = 1.0 / k
        time_factor = np.sqrt(np.maximum((T - t) / T, 0.0))
        alpha = 0.5 / (k * Q) if Q > 0 else 0.0
        bid_depth = d_base + alpha * q * time_factor
        ask_depth = d_base - alpha * q * time_factor
        min_depth = np.maximum(0.01, -3.0/k + 0.01)
        bid_depth = np.maximum(bid_depth, min_depth)
        ask_depth = np.maximum(ask_depth, min_depth)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth

    bid_table, ask_table = _get_optimal_table(params)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    T = params['T']
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = np.clip(step_idx, 0, steps - 1)
    
    # Map inventory to table index
    inv_idx = inventory + Q  # inventory in [-Q, Q] -> [0, 2Q]
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[step_idx, inv_idx]
    ask_depths = ask_table[step_idx, inv_idx]
    
    # Ensure valid range: > -3/k, no NaN
    k = params['k']
    min_depth = -3.0 / k + 0.01
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, min_depth), np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, min_depth), np.inf)
    
    # Ensure correct shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64)
    ask_depths = np.asarray(ask_depths, dtype=np.float64)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
