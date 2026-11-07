"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_depth_cache = {}
_start_time = time.time()


def _solve_foc_depth(C, k, A, dt):
    """
    Solve the first-order condition for optimal depth given marginal cost C.
    
    FOC: (1 - exp(-L)) + (d - C) * k * L * exp(-L) = 0
    where L = A * exp(-k*d) * dt
    
    Rearranged: exp(L) - 1 = (C - d) * k * L
    
    If C <= 0, optimal depth is 0 (no fill is better than any fill at this cost).
    Otherwise, solve for d > 0 using Newton's method.
    """
    if C <= 0:
        return 0.0
    
    # Define f(d) = exp(L) - 1 - (C - d) * k * L where L = A*exp(-k*d)*dt
    # f'(d) = -k*L*exp(L) + k*L + (C-d)*k^2*L*exp(-k*d)*(-1) ... let me be careful
    # L = A*exp(-k*d)*dt
    # dL/dd = -k*L
    # f(d) = exp(L) - 1 - (C-d)*k*L
    # f'(d) = exp(L)*dL/dd - k*d((C-d)*L)/dd
    #       = exp(L)*(-k*L) - k*(-L + (C-d)*dL/dd)
    #       = -k*L*exp(L) - k*(-L + (C-d)*(-k*L))
    #       = -k*L*exp(L) + k*L + k^2*L*(C-d)
    #       = k*L * (-exp(L) + 1 + k*(C-d))
    
    # Initial guess: d0 = C + 1/k (from small-L approximation)
    d = max(C + 1.0 / k, 0.01)
    
    for _ in range(30):
        L = A * np.exp(-k * d) * dt
        if L < 1e-15:
            return d
        f = np.exp(L) - 1.0 - (C - d) * k * L
        fp = k * L * (-np.exp(L) + 1.0 + k * (C - d))
        if abs(fp) < 1e-15:
            break
        d_new = d - f / fp
        if d_new < 0:
            d_new = 0.0
        if d_new > 50.0 / k:
            d_new = 50.0 / k
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    
    return max(d, 0.0)


def _solve_optimal_depths(sigma, A, k, gamma, Q, steps, T=1.0):
    """
    Solve for optimal bid/ask depths using backward induction with quadratic value function.
    """
    dt = T / steps
    nq = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Backward induction: V_{n}(q) = max_depths { immediate_reward + E[V_{n+1}(q_next)] }
    # We'll use a quadratic approximation V(q) = a*q^2 + b*q + c at each step
    # But for efficiency, we'll iterate on (a, b) assuming the structure is stable.
    
    # Initialize value function coefficients
    a = gamma * sigma**2 * T / 2.0
    b = 0.0
    
    # Iterate to convergence of (a, b)
    for iteration in range(200):
        # Compute optimal depths for all q using current (a, b)
        bid_depths = np.zeros(nq)
        ask_depths = np.zeros(nq)
        
        for i in range(nq):
            q = i - Q
            # Marginal cost of bid fill: V(q+1) - V(q) = a*(2q+1) + b
            C_b = a * (2 * q + 1) + b
            # Marginal cost of ask fill: V(q-1) - V(q) = -a*(2q-1) + b
            C_a = -a * (2 * q - 1) + b
            
            d_b = _solve_foc_depth(C_b, k, A, dt)
            d_a = _solve_foc_depth(C_a, k, A, dt)
            
            bid_depths[i] = d_b
            ask_depths[i] = d_a
        
        # Now update (a, b) using the Bellman equation
        # For each q, the Bellman equation gives:
        # V(q) = -(gamma/2)*sigma^2*q^2*dt + d_b*p_b + d_a*p_a + E[V(q_next)]
        # where E[V(q_next)] uses the NEW value function (next step)
        
        # We want to find (a_new, b_new) such that V_new(q) = a_new*q^2 + b_new*q + c_new
        # satisfies the Bellman equation for all q.
        
        # Build the system: for each q, 
        # a_new*q^2 + b_new*q + c_new = -(gamma/2)*sigma^2*q^2*dt + d_b(q)*p_b(q) + d_a(q)*p_a(q) 
        #   + p_b(q)*(1-p_a(q))*(a*(q+1)^2 + b*(q+1) + c)
        #   + (1-p_b(q))*p_a(q)*(a*(q-1)^2 + b*(q-1) + c)
        #   + (1 - p_b(q)*(1-p_a(q)) - (1-p_b(q))*p_a(q))*(a*q^2 + b*q + c)
        
        # Simplify the expectation term:
        # E[V] = a*E[q_next^2] + b*E[q_next] + c
        # E[q_next] = q + p_b*(1-p_a) - (1-p_b)*p_a
        # E[q_next^2] = E[(q + delta)^2] where delta is {-1, 0, +1}
        # = q^2 + 2q*(p_b*(1-p_a) - (1-p_b)*p_a) + p_b*(1-p_a) + (1-p_b)*p_a
        
        # So: a_new*q^2 + b_new*q + c_new = -(gamma/2)*sigma^2*q^2*dt + d_b*p_b + d_a*p_a
        #   + a*(q^2 + 2q*mu_q + var_q) + b*(q + mu_q) + c
        # where mu_q = p_b*(1-p_a) - (1-p_b)*p_a
        #       var_q = p_b*(1-p_a) + (1-p_b)*p_a
        
        # = a*q^2 + (b + 2*a*mu_q)*q + (a*var_q + b*mu_q + c)
        #   -(gamma/2)*sigma^2*q^2*dt + d_b*p_b + d_a*p_a
        
        # So: a_new*q^2 + b_new*q + c_new = (a - gamma*sigma^2*dt/2)*q^2 + (b + 2*a*mu_q)*q 
        #   + (a*var_q + b*mu_q + c + d_b*p_b + d_a*p_a)
        
        # For this to hold for all q with a quadratic, we need:
        # a_new = a - gamma*sigma^2*dt/2  ... but wait, mu_q and var_q depend on q!
        # This means the value function is NOT exactly quadratic. We need to fit.
        
        # Let's do a least-squares fit: for each q, compute the RHS and fit a_new*q^2 + b_new*q + c_new.
        
        # Compute RHS for all q
        rhs = np.zeros(nq)
        for i in range(nq):
            q = i - Q
            d_b = bid_depths[i]
            d_a = ask_depths[i]
            L_b = A * np.exp(-k * d_b) * dt
            L_a = A * np.exp(-k * d_a) * dt
            p_b = 1.0 - np.exp(-L_b)
            p_a = 1.0 - np.exp(-L_a)
            
            mu_q = p_b * (1 - p_a) - (1 - p_b) * p_a
            var_q = p_b * (1 - p_a) + (1 - p_b) * p_a
            
            rhs[i] = (a - gamma * sigma**2 * dt / 2.0) * q**2 + (b + 2 * a * mu_q) * q \
                     + (a * var_q + b * mu_q + d_b * p_b + d_a * p_a)
        
        # Fit: rhs ≈ a_new*q^2 + b_new*q + c_new
        # Use least squares
        A_mat = np.column_stack([q_vals**2, q_vals, np.ones(nq)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
        a_new, b_new, c_new = coeffs[0], coeffs[1], coeffs[2]
        
        # Check convergence
        if abs(a_new - a) < 1e-10 and abs(b_new - b) < 1e-10:
            a, b = a_new, b_new
            break
        
        a, b = a_new, b_new
    
    # Final computation of optimal depths
    bid_depths = np.zeros(nq)
    ask_depths = np.zeros(nq)
    for i in range(nq):
        q = i - Q
        C_b = a * (2 * q + 1) + b
        C_a = -a * (2 * q - 1) + b
        
        d_b = _solve_foc_depth(C_b, k, A, dt)
        d_a = _solve_foc_depth(C_a, k, A, dt)
        
        bid_depths[i] = d_b
        ask_depths[i] = d_a
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Inventory-aware quoting with precomputed optimal depths."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    steps = params["steps"]
    T = params["T"]
    
    # Deadline guard
    if time.time() - _start_time > 290:
        default_depth = 1.0 / k
        bid_depths = np.full(4000, default_depth, dtype=np.float64)
        ask_depths = np.full(4000, default_depth, dtype=np.float64)
        bid_depths[inventory >= Q] = np.inf
        ask_depths[inventory <= -Q] = np.inf
        return bid_depths, ask_depths
    
    # Check cache
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _depth_cache:
        _depth_cache[key] = _solve_optimal_depths(sigma, A, k, gamma, Q, steps, T)
    
    bid_depths_table, ask_depths_table = _depth_cache[key]
    
    # Map inventory to index
    q = inventory
    indices = (q + Q).astype(np.int64)
    indices = np.clip(indices, 0, 2 * Q)
    
    bid_depths = bid_depths_table[indices].copy()
    ask_depths = ask_depths_table[indices].copy()
    
    # Enforce no bid fill at q = Q, no ask fill at q = -Q
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    # Ensure valid values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 0.0, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 0.0, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
