# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 315.0  # Safety margin for 330s limit

def _compute_optimal_policy(params):
    """
    Compute optimal bid and ask depths using DP.
    Uses analytical gradient ascent on the continuous depth space instead of grid search.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    V_next = np.zeros(n_states)
    
    # Precompute cost for each state
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt  # (n_states,)
    
    # Depth bounds: [0, d_max]
    d_min = 0.0
    d_max = 3.0  # Corresponds to -3/k bound? No, bound is -3/k. Let's check.
    # "Values below -3/k ... make the whole result invalid"
    # Wait, the prompt says "distances below and above mid".
    # "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # This implies depth must be >= -3/k?
    # Usually depth is positive.
    # If bid_depth = d, bid price = mid - d.
    # The constraint "depth < -3/k" being invalid suggests depth should be > -3/k.
    # But typically depth is non-negative.
    # Let's assume depth is in [0, d_max] where d_max is reasonable.
    # Let's use d_max = 10/k to be safe, or just 5.0.
    # Let's stick to a reasonable range.
    d_max = 5.0
    
    # Iterate t from steps-1 down to 0
    for t in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _start_time > TIME_LIMIT:
            # Fallback: use zero depths (no quote) or previous values
            # Zero depth means quoting at mid.
            # Let's just break and fill remaining with 0
            for tt in range(t, -1, -1):
                opt_bid[tt, :] = 0.0
                opt_ask[tt, :] = 0.0
            break
        
        # v_q, v_qp, v_qm for all states
        v_q = V_next  # (n_states,)
        
        # v_qp: V_next[i+1] for i < n_states-1, else 0 (no bid at q=Q)
        v_qp = np.zeros(n_states)
        v_qp[:-1] = V_next[1:]
        
        # v_qm: V_next[i-1] for i > 0, else 0 (no ask at q=-Q)
        v_qm = np.zeros(n_states)
        v_qm[1:] = V_next[:-1]
        
        # Initial guess for depths: 0.0
        d_b = np.zeros(n_states)
        d_a = np.zeros(n_states)
        
        # Gradient ascent iterations
        # Objective J(d_b, d_a) = E[reward] - cost
        # E[reward] = (1-pb)(1-pa) v_q + pb(1-pa)(v_qp + d_b) + (1-pb)pa(v_qm + d_a) + pb pa (v_q + d_b + d_a)
        
        n_iter = 20
        step_size = 0.05
        
        for _ in range(n_iter):
            # Compute probabilities
            pb = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
            pa = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
            
            # Compute Expected Value
            v_no_fill = v_q
            v_bid_fill = v_qp + d_b
            v_ask_fill = v_qm + d_a
            v_both_fill = v_q + d_b + d_a
            
            EV = ((1-pb)*(1-pa)*v_no_fill +
                  pb*(1-pa)*v_bid_fill +
                  (1-pb)*pa*v_ask_fill +
                  pb*pa*v_both_fill)
            
            # Gradient of EV w.r.t d_b
            # dpb/dd_b = A * dt * exp(-k d_b) * (-k) * exp(-k d_b) ? No.
            # p = 1 - exp(-C), C = A exp(-k d) dt
            # dp/dd = -exp(-C) * dC/dd
            # dC/dd = A dt * (-k) exp(-k d)
            # dp/dd = -exp(-C) * (-k A dt exp(-k d)) = k A dt exp(-k d) exp(-C)
            
            exp_Cb = np.exp(-A * np.exp(-k * d_b) * dt)
            exp_Ca = np.exp(-A * np.exp(-k * d_a) * dt)
            
            dpb_ddb = k * A * dt * np.exp(-k * d_b) * exp_Cb
            dpa_dda = k * A * dt * np.exp(-k * d_a) * exp_Ca
            
            # dEV/dd_b = (1-pa) * d/dd_b [ pb(v_qp + d_b) + pb pa (v_q + d_b + d_a) ] - pb * d/dd_b [ (1-pb)(v_no) ... ]
            # Let's use the full derivative expression.
            # EV = (1-pb)(1-pa)v_q + pb(1-pa)(v_qp + d_b) + (1-pb)pa(v_qm + d_a) + pb pa (v_q + d_b + d_a)
            
            # dEV/dd_b:
            # Term 1: -dpb (1-pa) v_q
            # Term 2: dpb (1-pa)(v_qp + d_b) + pb (1-pa) * 1
            # Term 3: -dpb pa (v_qm + d_a)
            # Term 4: dpb pa (v_q + d_b + d_a) + pb pa * 1
            
            dEV_ddb = (
                -dpb_ddb * (1-pa) * v_q +
                dpb_ddb * (1-pa) * (v_qp + d_b) + pb * (1-pa) +
                -dpb_ddb * pa * (v_qm + d_a) +
                dpb_ddb * pa * (v_q + d_b + d_a) + pb * pa
            )
            
            # dEV/dd_a:
            dEV_dda = (
                -(1-pb) * dpa_dda * v_q +
                (1-pb) * dpa_dda * (v_qm + d_a) + (1-pb) * pa +
                pb * dpa_dda * (v_q + d_b + d_a) + pb * dpa_dda * 0 + pa * pb  # Wait, term 3 has (1-pb)pa(v_qm+d_a). d/da of (1-pb) is 0. d/da of pa is dpa. d/da of (v_qm+d_a) is 1.
                # Let's re-verify Term 3 derivative: (1-pb) * d/da [ pa (v_qm + d_a) ] = (1-pb) [ dpa (v_qm+d_a) + pa * 1 ]
                # My previous line: (1-pb) * dpa_dda * (v_qm + d_a) + (1-pb) * pa. Correct.
                # Term 4: d/da [ pb pa (v_q + d_b + d_a) ] = pb [ dpa (v_q+d_b+d_a) + pa * 1 ]
                # My previous line: pb * dpa_dda * (v_q + d_b + d_a) + pa * pb. Correct.
            )
            
            # Update
            d_b += step_size * dEV_ddb
            d_a += step_size * dEV_dda
            
            # Clip
            d_b = np.clip(d_b, d_min, d_max)
            d_a = np.clip(d_a, d_min, d_max)
        
        opt_bid[t, :] = d_b
        opt_ask[t, :] = d_a
        
        # Update V_next
        pb = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
        pa = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
        
        v_no_fill = v_q
        v_bid_fill = v_qp + d_b
        v_ask_fill = v_qm + d_a
        v_both_fill = v_q + d_b + d_a
        
        V_next = ((1-pb)*(1-pa)*v_no_fill +
                  pb*(1-pa)*v_bid_fill +
                  (1-pb)*pa*v_ask_fill +
                  pb*pa*v_both_fill) - cost_per_state
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
