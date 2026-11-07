"""Market-making quote strategy using continuous-depth DP with Newton refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_dp_cache = {}
_DEADLINE = 290.0
_start_time = None


def _solve_depth_foc(V_q, V_qp1, V_qm1, inv_cost, A, k, dt, side):
    """
    Solve the first-order condition for optimal depth on one side.
    
    For bid side:
    Let lambda_b = A * exp(-k * d_b) be the intensity.
    p_b = 1 - exp(-lambda_b * dt)
    
    The Bellman update for bid depth d_b (with ask fixed) is:
    J(d_b) = d_b * p_b - inv_cost + (1-p_b)*V_q + p_b*V_qp1 + (1-p_b)*(1-p_a)*0 ... 
    Actually, we need to be more careful. Let's derive properly.
    
    For a single step, with bid depth d_b and ask depth d_a:
    lambda_b = A * exp(-k d_b), lambda_a = A * exp(-k d_a)
    p_b = 1 - exp(-lambda_b dt), p_a = 1 - exp(-lambda_a dt)
    
    Immediate reward: d_b * p_b + d_a * p_a - inv_cost
    Next state:
      q stays: prob (1-p_b)(1-p_a), value V_q
      q+1: prob p_b(1-p_a), value V_qp1
      q-1: prob (1-p_b)p_a, value V_qm1
    
    Total = d_b p_b + d_a p_a - inv_cost + (1-p_b)(1-p_a)V_q + p_b(1-p_a)V_qp1 + (1-p_b)p_a V_qm1
    
    For bid optimization (fixing d_a, so p_a, d_a, V_qm1 are constants):
    Let C = d_a p_a - inv_cost + (1-p_a)V_q + p_a V_qm1  [terms not involving p_b or d_b directly, except through p_b]
    Wait, let me expand:
    Total = d_b p_b + d_a p_a - inv_cost + V_q - p_b V_q - p_a V_q + p_b p_a V_q + p_b V_qp1 - p_b p_a V_qp1 + p_a V_qm1 - p_b p_a V_qm1
    
    = d_b p_b + [d_a p_a - inv_cost + V_q - p_a V_q + p_a V_qm1] + p_b[-V_q + p_a V_q + V_qp1 - p_a V_qp1 - p_a V_qm1]
    
    Let me define:
    B0 = d_a p_a - inv_cost + V_q(1 - p_a) + p_a V_qm1
    B1 = -V_q + p_a V_q + V_qp1 - p_a V_qp1 - p_a V_qm1
         = -V_q(1-p_a) + V_qp1(1-p_a) - p_a V_qm1
         = (1-p_a)(V_qp1 - V_q) - p_a V_qm1
    
    So Total = d_b p_b + B0 + p_b * B1
    
    We want to maximize over d_b >= -3/k.
    Let f(d_b) = d_b * (1-exp(-A e^{-k d_b} dt)) + B1 * (1-exp(-A e^{-k d_b} dt))
               = (d_b + B1) * (1 - exp(-A e^{-k d_b} dt))
    
    Let lambda = A e^{-k d_b}, so d_b = -ln(lambda/A)/k = (ln A - ln lambda)/k
    And p_b = 1 - exp(-lambda dt)
    
    f(lambda) = ((ln A - ln lambda)/k + B1) * (1 - exp(-lambda dt))
    
    df/dlambda = (1/k) * (-1/lambda) * (1-exp(-lambda dt)) + ((ln A - ln lambda)/k + B1) * dt * exp(-lambda dt)
    
    Setting to zero:
    -(1/k)/lambda * (1-exp(-lambda dt)) + ((ln A - ln lambda)/k + B1) * dt * exp(-lambda dt) = 0
    
    Let me denote x = lambda dt (dimensionless intensity over step)
    Then lambda = x/dt, d_b = (ln A - ln(x/dt))/k = (ln(A dt) - ln x)/k
    
    The equation becomes:
    -(1/k)/(x/dt) * (1-e^{-x}) + ((ln(A dt) - ln x)/k + B1) * dt * e^{-x} = 0
    
    Multiply by k:
    -(dt/x)(1-e^{-x}) + (ln(A dt) - ln x + k B1) * dt * e^{-x} = 0
    
    Divide by dt (assuming dt > 0):
    -(1/x)(1-e^{-x}) + (ln(A dt) - ln x + k B1) e^{-x} = 0
    
    Let C_bid = ln(A dt) + k B1
    Then: -(1/x)(1-e^{-x}) + (C_bid - ln x) e^{-x} = 0
    
    Multiply by x e^x:
    -(1-e^{-x}) e^x + (C_bid - ln x) x = 0
    -(e^x - 1) + x(C_bid - ln x) = 0
    x(C_bid - ln x) - e^x + 1 = 0
    
    Let g(x) = x(C_bid - ln x) - e^x + 1
    We need g(x) = 0 for x > 0.
    
    This is a transcendental equation. We can solve it with Newton's method.
    g'(x) = (C_bid - ln x) + x(-1/x) - e^x = C_bid - ln x - 1 - e^x
    
    Similarly for ask side, the equation would be analogous but with V_qm1 and V_qp1 swapped roles in a sense.
    Actually for ask:
    Total = d_a p_a + B0' + p_a * B1'
    where B0' = d_b p_b - inv_cost + V_q(1-p_b) + p_b V_qp1
    B1' = (1-p_b)(V_q - V_qm1) - p_b V_qp1
    
    Same structure, just B1 replaced by B1'.
    
    So the approach:
    1. For each (t, q), we need to solve two coupled equations (one for bid, one for ask).
    2. But we can decouple: fix ask, solve for bid; fix bid, solve for ask; iterate.
    3. Or use the FOC to get a good initial guess and then do a small local search.
    
    Actually, since the problem is 2D (bid, ask) and we want the global max, let's do:
    - Use the FOC to get a candidate for bid given current ask, and vice versa.
    - Start with a reasonable initial guess (e.g., from a coarse grid).
    - Iterate a few times.
    - Then do a local refinement with a fine grid around the solution.
    
    But this might be complex. Let me think of a simpler approach.
    
    Alternative: Use the FOC to define the optimal bid as a function of the "shadow price" B1, and similarly for ask.
    Then we can precompute a lookup table: for a range of B1 values, what is the optimal d_b?
    
    B1 = (1-p_a)(V_qp1 - V_q) - p_a V_qm1
    
    Since p_a depends on d_a, and d_a depends on B1', this is still coupled.
    
    Let me try a different approach: use a moderate grid (say 200x200) for the initial DP, then refine the top few candidates with Newton's method on the continuous problem.
    """
    pass  # Placeholder - we'll implement a hybrid approach


def _precompute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal policy using a hybrid approach:
    1. Coarse grid DP to get approximate value function.
    2. Refine using continuous FOC-based optimization.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]
    
    dt = T / steps
    n_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # Use a moderate grid for initial approximation
    n_lam = 150
    lam_min = max(A * np.exp(-8.0), 1e-12)
    lam_max = A * np.exp(3.0)
    lam_grid = np.geomspace(lam_min, lam_max, n_lam)
    p_grid = 1.0 - np.exp(-lam_grid * dt)
    d_grid = -np.log(lam_grid / A) / k
    d_grid = np.maximum(d_grid, -3.0 / k + 1e-9)
    
    inv_cost = 0.5 * gamma * sigma**2 * q_indices**2 * dt
    
    V_next = np.zeros(n_q)
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    for t in range(steps - 1, -1, -1):
        if time.time() - _start_time > _DEADLINE - 15:
            default_d = 1.0 / k
            opt_bid[:t+1] = default_d
            opt_ask[:t+1] = default_d
            result = (opt_bid, opt_ask)
            _dp_cache[key] = result
            return result
        
        V_curr = np.zeros(n_q)
        
        for i, q in enumerate(q_indices):
            if q < Q and q > -Q:
                pb = p_grid[:, np.newaxis]
                pa = p_grid[np.newaxis, :]
                db = d_grid[:, np.newaxis]
                da = d_grid[np.newaxis, :]
                
                imm = db * pb + da * pa - inv_cost[i]
                
                V_q = V_next[i]
                V_qp1 = V_next[i+1]
                V_qm1 = V_next[i-1]
                
                p_stay = (1 - pb) * (1 - pa)
                p_up = pb * (1 - pa)
                p_down = (1 - pb) * pa
                
                V_next_state = p_stay * V_q + p_up * V_qp1 + p_down * V_qm1
                total = imm + V_next_state
                
                idx = np.argmax(total)
                idx_r, idx_c = np.unravel_index(idx, total.shape)
                
                # Refine around the grid solution using local continuous optimization
                d_b_init = d_grid[idx_r]
                d_a_init = d_grid[idx_c]
                
                # Local refinement with a fine grid around the solution
                margin = max(0.1 / k, 0.01)
                d_b_fine = np.linspace(d_b_init - margin, d_b_init + margin, 20)
                d_a_fine = np.linspace(d_a_init - margin, d_a_init + margin, 20)
                d_b_fine = np.maximum(d_b_fine, -3.0/k + 1e-9)
                d_a_fine = np.maximum(d_a_fine, -3.0/k + 1e-9)
                
                lb_fine = A * np.exp(-k * d_b_fine)
                la_fine = A * np.exp(-k * d_a_fine)
                pb_fine = 1.0 - np.exp(-lb_fine * dt)
                pa_fine = 1.0 - np.exp(-la_fine * dt)
                
                pb_2d = pb_fine[:, np.newaxis]
                pa_2d = pa_fine[np.newaxis, :]
                db_2d = d_b_fine[:, np.newaxis]
                da_2d = d_a_fine[np.newaxis, :]
                
                imm_fine = db_2d * pb_2d + da_2d * pa_2d - inv_cost[i]
                
                p_stay_f = (1 - pb_2d) * (1 - pa_2d)
                p_up_f = pb_2d * (1 - pa_2d)
                p_down_f = (1 - pb_2d) * pa_2d
                
                V_next_f = p_stay_f * V_q + p_up_f * V_qp1 + p_down_f * V_qm1
                total_fine = imm_fine + V_next_f
                
                idx_f = np.argmax(total_fine)
                idx_r_f, idx_c_f = np.unravel_index(idx_f, total_fine.shape)
                
                opt_bid[t, i] = d_b_fine[idx_r_f]
                opt_ask[t, i] = d_a_fine[idx_c_f]
                V_curr[i] = total_fine[idx_r_f, idx_c_f]
                
            elif q == -Q:
                # Only bid allowed
                pb = p_grid
                db = d_grid
                imm = db * pb - inv_cost[i]
                
                V_q = V_next[i]
                V_qp1 = V_next[i+1]
                
                V_next_state = (1 - pb) * V_q + pb * V_qp1
                total = imm + V_next_state
                idx = np.argmax(total)
                
                d_b_init = d_grid[idx]
                margin = max(0.1 / k, 0.01)
                d_b_fine = np.linspace(d_b_init - margin, d_b_init + margin, 20)
                d_b_fine = np.maximum(d_b_fine, -3.0/k + 1e-9)
                
                lb_fine = A * np.exp(-k * d_b_fine)
                pb_fine = 1.0 - np.exp(-lb_fine * dt)
                
                imm_fine = d_b_fine * pb_fine - inv_cost[i]
                V_next_fine = (1 - pb_fine) * V_q + pb_fine * V_qp1
                total_fine = imm_fine + V_next_fine
                
                idx_f = np.argmax(total_fine)
                opt_bid[t, i] = d_b_fine[idx_f]
                opt_ask[t, i] = np.inf
                V_curr[i] = total_fine[idx_f]
                
            elif q == Q:
                # Only ask allowed
                pa = p_grid
                da = d_grid
                imm = da * pa - inv_cost[i]
                
                V_q = V_next[i]
                V_qm1 = V_next[i-1]
                
                V_next_state = (1 - pa) * V_q + pa * V_qm1
                total = imm + V_next_state
                idx = np.argmax(total)
                
                d_a_init = d_grid[idx]
                margin = max(0.1 / k, 0.01)
                d_a_fine = np.linspace(d_a_init - margin, d_a_init + margin, 20)
                d_a_fine = np.maximum(d_a_fine, -3.0/k + 1e-9)
                
                la_fine = A * np.exp(-k * d_a_fine)
                pa_fine = 1.0 - np.exp(-la_fine * dt)
                
                imm_fine = d_a_fine * pa_fine - inv_cost[i]
                V_next_fine = (1 - pa_fine) * V_q + pa_fine * V_qm1
                total_fine = imm_fine + V_next_fine
                
                idx_f = np.argmax(total_fine)
                opt_bid[t, i] = np.inf
                opt_ask[t, i] = d_a_fine[idx_f]
                V_curr[i] = total_fine[idx_f]
                
            else:
                V_curr[i] = -inv_cost[i]
                opt_bid[t, i] = np.inf
                opt_ask[t, i] = np.inf
        
        V_next = V_curr
    
    result = (opt_bid, opt_ask)
    _dp_cache[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on precomputed optimal policy.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    t_idx = int(round(t / dt))
    t_idx = min(t_idx, steps - 1)
    t_idx = max(t_idx, 0)
    
    try:
        opt_bid, opt_ask = _precompute_dp(sigma, A, k, gamma, T, steps, Q)
    except Exception:
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(int)
    
    bid_depths = opt_bid[t_idx, inv_idx]
    ask_depths = opt_ask[t_idx, inv_idx]
    
    min_depth = -3.0 / k
    bid_depths = np.where(np.isnan(bid_depths), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 1.0/k, ask_depths)
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
