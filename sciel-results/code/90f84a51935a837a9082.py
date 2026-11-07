"""
Strategy: Continuous Policy Optimization via Gradient Descent.
Instead of discretizing the depth space into a grid (which limits precision to grid spacing),
we parameterize the optimal bid and ask depth for each inventory state q as continuous variables.
We initialize these parameters heuristically and refine them using numerical gradient ascent
on the expected immediate reward plus discounted future value (approximated via a few steps of rollout or stationary assumption).

This approach:
1. Avoids the O(N_grid^2) memory blowup of the 3D DP table.
2. Allows depths to take any real value, potentially finding the exact optimal depth.
3. Uses a much smaller state space (61 parameters per side) to optimize.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin before 330s kill
_policy_cache = {}

def _compute_continuous_policy(params):
    """
    Compute optimal bid and ask depths for all inventory states using gradient ascent.
    We treat the problem as finding the stationary policy that maximizes the Bellman optimality equation's RHS.
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
    
    # Initialize depths. 
    # Heuristic: In market making, wider spreads when inventory is high (to avoid further risk) or low (to build position).
    # Actually, usually you skew quotes. 
    # Let's start with a reasonable base spread, e.g., 0.5.
    # bid_depth and ask_depth are distances from mid.
    # bid = mid - bid_depth, ask = mid + ask_depth.
    
    # We optimize bid_depths and ask_depths for each q.
    # Shape: (n_states,)
    init_depth = 0.5
    bid_depths = np.full(n_states, init_depth)
    ask_depths = np.full(n_states, init_depth)
    
    # Precompute cost
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # We will perform gradient ascent on the "Value Function" approximation.
    # V(q) = max_d (Reward + E[V(q')])
    # Since we don't have V, we can use a few iterations of Value Iteration where the "action" is continuous.
    # Alternatively, we can just optimize the immediate reward + a heuristic future term.
    # Better: Use the fact that the optimal policy satisfies the Bellman equation.
    # We can iterate:
    # 1. Given current V, find optimal d* by maximizing Q(q, d; V).
    # 2. Update V using d*.
    # This is equivalent to policy iteration with continuous actions.
    
    V = np.zeros(n_states)
    
    # Helper to compute Q-values for a given depth array and V
    def compute_q_values(bid_d, ask_d, V):
        # bid_d, ask_d: (n_states,)
        # V: (n_states,)
        
        # Probabilities
        p_b = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
        
        # Next state values
        # q -> q+1 (bid fill)
        v_qp = np.zeros(n_states)
        v_qp[:-1] = V[1:]
        
        # q -> q-1 (ask fill)
        v_qm = np.zeros(n_states)
        v_qm[1:] = V[:-1]
        
        # Rewards
        r_b = bid_d
        r_a = ask_d
        
        # Expected Value
        # No fill: (1-pb)(1-pa) * V(q)
        # Bid fill: pb(1-pa) * (V(q+1) + r_b)
        # Ask fill: (1-pb)pa * (V(q-1) + r_a)
        # Both: pb pa * (V(q) + r_b + r_a)
        
        q_val = (1-pb)*(1-pa) * V + \
                pb*(1-pa) * (v_qp + r_b) + \
                (1-pb)*pa * (v_qm + r_a) + \
                pb*pa * (V + r_b + r_a)
        
        # Subtract holding cost
        q_val -= cost_per_state
        
        return q_val, p_b, p_a, v_qp, v_qm, r_b, r_a

    # Policy Iteration
    max_iter = 100
    tol = 1e-5
    
    for it in range(max_iter):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        # 1. Policy Evaluation / Update
        # We need to find optimal bid_d and ask_d for each state independently given V.
        # This is a 1D optimization problem for each state.
        # We can use a small grid search or gradient step for the action selection.
        # Since we want continuous, let's use a numerical gradient step or a fine grid.
        # A fine grid of 100 points per state is cheap: 61 * 100 = 6100 evaluations.
        
        new_bid_depths = np.zeros(n_states)
        new_ask_depths = np.zeros(n_states)
        
        # Use a grid for action selection to be robust, then refine?
        # Or just use a grid. The previous strategy failed due to *global* grid resolution.
        # Here, we can use a high-resolution grid locally for action selection, but V is updated via exact expectation.
        # Actually, if we use a grid for action selection, we are back to grid-based.
        # Let's use gradient ascent on the action.
        
        # Initialize action with current policy
        bid_d = bid_depths.copy()
        ask_d = ask_depths.copy()
        
        # Perform a few steps of gradient ascent on the action for each state
        # We can vectorize the gradient computation.
        
        # Let's define a function to compute gradient of Q w.r.t depth.
        # Q(q, db, da) = ...
        # dQ/dbid = r_b + ... - cost_derived? No, cost is independent of action.
        # The term is:
        # Q = (1-pb)(1-pa)V + pb(1-pa)(V_qp + db) + (1-pb)pa(V_qm + da) + pb pa (V + db + da) - C
        # dQ/db = 
        #   -dpb/db (1-pa) V + dpb/db (1-pa)(V_qp + db) + pb (-dpa/da? no, da is independent) 
        #   Wait, pb depends on db, pa depends on da.
        #   dQ/db = dpb/db [ (1-pa)(V_qp + db) - (1-pa)V + pa(V + db + da) - pa V ] + (1-pa) + pa
        #   Simplify:
        #   Term1 (no fill): -dpb/db (1-pa) V
        #   Term2 (bid fill): dpb/db (1-pa) (V_qp + db) + (1-pa) * 1
        #   Term3 (ask fill): 0 (pa doesn't depend on db, V_qm, da don't depend on db)
        #   Term4 (both): dpb/db [ pa(V + db + da) - pa V ] + pa * 1
        #   = dpb/db [ pa db ] + pa
        #   
        #   Sum: dpb/db { (1-pa)(V_qp + db) - (1-pa)V + pa db } + (1-pa) + pa
        #   = dpb/db { (1-pa)(V_qp - V) + db } + 1
        
        # Similarly for da:
        # dQ/da = dpa/da { (1-pb)(V_qm - V) + da } + 1
        
        # dpb/db = -A exp(-k db) dt exp(-k db) ? No.
        # p = 1 - exp(-A exp(-k d) dt)
        # dp/dd = -exp(...) * d/dd (-A exp(-k d) dt) = exp(-A exp(-k d) dt) * A exp(-k d) dt * k
        # Let E = exp(-k d)
        # p = 1 - exp(-A E dt)
        # dp/dd = exp(-A E dt) * A E dt * k
        
        for step in range(10): # 10 refinement steps
            if time.time() - _start_time > TIME_LIMIT:
                break
            
            p_b, p_a = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt), 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
            
            v_qp = np.zeros(n_states); v_qp[:-1] = V[1:]
            v_qm = np.zeros(n_states); v_qm[1:] = V[:-1]
            
            # Gradients
            # dpb/db
            exp_term_b = np.exp(-k * bid_d)
            dpb_db = np.exp(-A * exp_term_b * dt) * A * exp_term_b * dt * k
            dpa_da = np.exp(-A * np.exp(-k * ask_d) * dt) * A * np.exp(-k * ask_d) * dt * k
            
            grad_b = dpb_db * ((1 - p_a) * (v_qp - V) + bid_d) + 1.0
            grad_a = dpa_da * ((1 - p_b) * (v_qm - V) + ask_d) + 1.0
            
            # Learning rate
            lr = 0.05
            bid_d += lr * grad_b
            ask_d += lr * grad_a
            
            # Clip to valid range [0, 3/k] approx? No, just [0, inf).
            # Problem says invalid if < -3/k. So 0 is safe.
            bid_d = np.maximum(bid_d, 0.0)
            ask_d = np.maximum(ask_d, 0.0)
            
            # Optional: Check for convergence or oscillation
            # If gradients are small, stop early
            
        bid_depths = bid_d
        ask_depths = ask_d
        
        # 2. Value Update
        q_val, _, _, _, _, _, _ = compute_q_values(bid_depths, ask_depths, V)
        
        # V_new = max over actions? We just used the greedy action.
        # In policy iteration, V_new is the value of the new policy.
        # We approximate it by the Q-value of the chosen action.
        # This is a single-step approximation. To be more accurate, we could iterate V until convergence for this policy.
        # But that's expensive. Let's just do V = q_val for now (one-step TD update essentially).
        
        diff = np.max(np.abs(q_val - V))
        V = q_val
        
        if diff < tol:
            break
            
    # Construct full policy arrays
    opt_bid = np.tile(bid_depths[None, :], (steps, 1))
    opt_ask = np.tile(ask_depths[None, :], (steps, 1))
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed continuous policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_continuous_policy(params)
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
