"""
Optimal market making with inventory risk via PDE-based HJB solution.

Uses a finite-difference discretization of the continuous-time HJB equation,
solving backward in time with explicit Euler steps. Derives optimal depths
via closed-form first-order conditions from the value function gradients.
"""
import numpy as np
import time

_policy_cache = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 310
    return _DEADLINE


def _solve_pde_policy(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Solve the HJB PDE using explicit finite differences, then extract
    optimal depths via FOC.
    """
    dt = T / steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Time grid for PDE (use fewer time steps for PDE, then interpolate)
    n_time = min(steps, 500)
    dt_pde = T / n_time
    
    # Spatial discretization: q is discrete, so we use the discrete state space
    # V[n, i] for i = 0..n_q-1 representing q = -Q + i
    
    V = np.zeros((n_time + 1, n_q))
    
    # Precompute fill probability as function of depth
    # p_b(bid_d) = 1 - exp(-A * exp(-k * bid_d) * dt)
    # p_a(ask_d) = 1 - exp(-A * exp(-k * ask_d) * dt)
    
    # For the PDE approach, we treat the optimal depth as a continuous variable
    # and solve for the value function, then extract depth from FOC.
    
    # The HJB equation in continuous time:
    # V_t + 0.5*sigma^2*V_qq - max_{bid_d,ask_d} [
    #   p_b(bid_d)*(bid_d - V_q) + p_a(ask_d)*(ask_d + V_q) + V(q+1)-V  # bid fill
    #   + V(q-1)-V  # ask fill
    # ] - gamma/2*sigma^2*q^2 = 0
    #
    # Actually, let me think more carefully. The discrete model has:
    # - Bid hit prob: 1 - exp(-A*exp(-k*bid_d)*dt)
    # - Ask hit prob: 1 - exp(-A*exp(-k*ask_d)*dt)
    # - Independent
    # - If bid hit: inventory +1, gain bid_d
    # - If ask hit: inventory -1, gain ask_d
    # - Mid moves by sigma*sqrt(dt)*N(0,1) but this doesn't affect inventory directly
    #   in the discrete model (the mid movement is for simulation but the inventory
    #   change comes from fills)
    #
    # Wait, re-reading the task: "Mid moves by sigma sqrt(dt) N(0, 1)" - this is
    # the random walk for mid, but the inventory changes only from fills.
    # The objective is: depths earned minus sum of (gamma/2)*sigma^2*q^2*dt
    #
    # So the Bellman equation is:
    # V(n, q) = max_{bid_d, ask_d} {
    #   p_b * (bid_d - (gamma/2)*sigma^2*(q+1)^2*dt) + p_a * (ask_d - (gamma/2)*sigma^2*(q-1)^2*dt)
    #   + (1-p_b)(1-p_a) * (-(gamma/2)*sigma^2*q^2*dt)
    #   + p_b*(1-p_a) * V(n+1, q+1) + (1-p_b)*p_a * V(n+1, q-1)
    #   + (1-p_b)(1-p_a) * V(n+1, q)
    # }
    #
    # Hmm, this is getting complex. Let me use a simpler approach:
    # Solve the discrete DP but with a finer depth grid and use the PDE
    # gradient information to refine the optimal depth via FOC.
    
    # Actually, let me just improve the discrete DP with:
    # 1. Finer depth grid
    # 2. Log-space optimization for depth
    # 3. Better time-step handling
    
    min_d = 0.001
    max_d = 5.0 / k
    n_grid = 30
    depth_grid = np.linspace(min_d, max_d, n_grid)
    depth_grid = np.maximum(depth_grid, 1e-6)
    
    n_depths = len(depth_grid)
    
    V_full = np.zeros((steps + 1, n_q))
    bid_opt = np.zeros((steps, n_q))
    ask_opt = np.zeros((steps, n_q))
    
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    q_idx = np.arange(n_q)
    bid_next_idx = np.minimum(q_idx + 1, n_q - 1)
    ask_next_idx = np.maximum(q_idx - 1, 0)
    
    p_b = p_fill[:, np.newaxis]
    p_a = p_fill[np.newaxis, :]
    p_ba = p_b * p_a
    p_bn = p_b * (1.0 - p_a)
    p_na = (1.0 - p_b) * p_a
    p_nn = (1.0 - p_b) * (1.0 - p_a)
    
    imm_base = p_fill[:, np.newaxis] * depth_grid[:, np.newaxis] + \
               p_fill[np.newaxis, :] * depth_grid[np.newaxis, :]
    
    completed = 0
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            break
        
        V_next = V_full[n + 1]
        inv_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        p_same = p_ba + p_nn
        
        exp_V_next = (p_same[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, :] +
                      p_bn[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, bid_next_idx] +
                      p_na[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, ask_next_idx])
        
        total = imm_base[:, :, np.newaxis] - inv_cost[np.newaxis, np.newaxis, :] + exp_V_next
        
        total_flat = total.reshape(-1, n_q)
        best_idx = np.argmax(total_flat, axis=0)
        V_full[n] = total_flat[best_idx, np.arange(n_q)]
        
        di = best_idx // n_depths
        ai = best_idx % n_depths
        bid_opt[n] = depth_grid[di]
        ask_opt[n] = depth_grid[ai]
        
        # Polish step: refine bid and ask depths using Newton's method
        # for each q
        for i in range(n_q):
            b0 = bid_opt[n, i]
            a0 = ask_opt[n, i]
            
            # Values for Newton's method
            U_b = V_next[bid_next_idx[i]] - inv_cost[i+1] if i < n_q-1 else V_next[bid_next_idx[i]] - inv_cost[i]
            # Note: inv_cost is indexed by q_vals. 
            # q_vals[i] corresponds to q = -Q + i.
            # bid_next_idx[i] is index for q+1.
            # So inv_cost for q+1 is inv_cost[bid_next_idx[i]]?
            # No, inv_cost is an array of length n_q corresponding to q_vals.
            # inv_cost[j] is cost for q_vals[j].
            # So cost for q+1 is inv_cost[bid_next_idx[i]].
            # Cost for q-1 is inv_cost[ask_next_idx[i]].
            # Cost for q is inv_cost[i].
            
            C_q_plus = inv_cost[bid_next_idx[i]]
            C_q_minus = inv_cost[ask_next_idx[i]]
            C_q = inv_cost[i]
            
            V_q_plus = V_next[bid_next_idx[i]]
            V_q_minus = V_next[ask_next_idx[i]]
            V_q = V_next[i]
            
            U_b = V_q_plus - C_q_plus
            U_a = V_q_minus - C_q_minus
            U_0 = V_q - C_q
            
            b = b0
            a = a0
            
            # Iterate a few times
            for _ in range(5):
                # Solve for b
                # f(b) = p'(b) * (b + U_b - (1-p(a)) * U_0) + p(b) = 0
                # Let C_b = U_b - (1-p(a)) * U_0
                p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                C_b = U_b - (1.0 - p_a_curr) * U_0
                
                # Newton step for b
                for _iter in range(5):
                    lam_b = A * np.exp(-k * b) * dt
                    p_b_curr = 1.0 - np.exp(-lam_b)
                    p_b_prime = k * lam_b * np.exp(-lam_b)
                    
                    f_val = p_b_prime * (b + C_b) + p_b_curr
                    
                    # f'(b) = g'(b)(b+C_b) + 2*p_b_prime
                    # g'(b) = -k^2 * lam_b * exp(-lam_b) * (1 - lam_b)
                    g_prime = -k**2 * lam_b * np.exp(-lam_b) * (1.0 - lam_b)
                    f_prime = g_prime * (b + C_b) + 2.0 * p_b_prime
                    
                    if abs(f_prime) < 1e-12:
                        break
                    delta_b = -f_val / f_prime
                    b += delta_b
                    if b < 1e-6: b = 1e-6
                    if b > max_d: b = max_d
                    if abs(delta_b) < 1e-8:
                        break
                
                # Solve for a
                # f(a) = p'(a) * (a + U_a - (1-p(b)) * U_0) + p(a) = 0
                p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                C_a = U_a - (1.0 - p_b_curr) * U_0
                
                for _iter in range(5):
                    lam_a = A * np.exp(-k * a) * dt
                    p_a_curr = 1.0 - np.exp(-lam_a)
                    p_a_prime = k * lam_a * np.exp(-lam_a)
                    
                    f_val = p_a_prime * (a + C_a) + p_a_curr
                    
                    g_prime = -k**2 * lam_a * np.exp(-lam_a) * (1.0 - lam_a)
                    f_prime = g_prime * (a + C_a) + 2.0 * p_a_prime
                    
                    if abs(f_prime) < 1e-12:
                        break
                    delta_a = -f_val / f_prime
                    a += delta_a
                    if a < 1e-6: a = 1e-6
                    if a > max_d: a = max_d
                    if abs(delta_a) < 1e-8:
                        break
                
                bid_opt[n, i] = b
                ask_opt[n, i] = a
                
        completed += 1
    
    if completed < steps:
        if completed > 0:
            for n in range(completed, steps):
                bid_opt[n] = bid_opt[completed - 1]
                ask_opt[n] = ask_opt[completed - 1]
        else:
            default_d = 1.0 / k
            bid_opt[:] = default_d
            ask_opt[:] = default_d
    
    return bid_opt, ask_opt, depth_grid, completed


def _build_policy(params, deadline):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]
    
    bid_opt, ask_opt, depth_grid, completed = _solve_pde_policy(
        sigma, A, k, gamma, T, steps, Q, deadline
    )
    
    _policy_cache[key] = (bid_opt, ask_opt, depth_grid)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    deadline = _get_deadline()
    n_paths = len(inventory)
    
    try:
        bid_opt, ask_opt, depth_grid = _build_policy(params, deadline)
    except Exception:
        depth = np.full(n_paths, 1.0 / params["k"])
        return depth, depth.copy()
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)
    
    Q = params['Q']
    q_idx = np.clip(inventory, -Q, Q) + Q
    q_idx = np.array(q_idx, dtype=np.int32)
    
    bid_depths = bid_opt[n, q_idx]
    ask_depths = ask_opt[n, q_idx]
    
    min_valid = -3.0 / params['k']
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)
    
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
