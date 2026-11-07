"""Market-making with inventory risk: Continuous DP with Analytical FOC."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit
_DP_CACHE = {}

def _solve_dp_continuous(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the DP using continuous optimization via First-Order Conditions.
    Returns optimal bid and ask depths for each step and inventory state.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    n_steps = steps
    n_states = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # V[t, q_idx] = value function at step t (t=0 is last step, t=n_steps is first)
    V = np.zeros((n_steps + 1, n_states))
    opt_bid = np.zeros((n_steps + 1, n_states))
    opt_ask = np.zeros((n_steps + 1, n_states))
    
    # Pre-compute constants
    # p(d) = 1 - exp(-A exp(-k d) dt)
    # Let lambda(d) = A exp(-k d) dt
    # p(d) = 1 - exp(-lambda(d))
    # dp/dd = -exp(-lambda) * (-lambda') = exp(-lambda) * lambda * k
    # lambda' = -k lambda
    # dp/dd = k * lambda * (1 - p)
    
    # We need to maximize:
    # J(b, a) = p_b b + p_a a - cost + E[V_next]
    # E[V_next] = V_q + p_b(V_{q+1}-V_q) + p_a(V_{q-1}-V_q) - p_b p_a (V_{q+1} + V_{q-1} - 2V_q)
    
    # For fixed a, maximize over b:
    # f(b) = p_b b + p_b (V_{q+1}-V_q) - p_b p_a (V_{q+1} + V_{q-1} - 2V_q) + const
    # Let D1 = V_{q+1} - V_q
    # Let D2 = V_{q-1} - V_q
    # Let D3 = V_{q+1} + V_{q-1} - 2V_q
    # f(b) = p_b (b + D1 - p_a D3) + const
    # Note: p_a is constant when optimizing b.
    # Let C_b = D1 - p_a D3
    # Maximize g(b) = p_b (b + C_b)
    # g'(b) = p_b' (b + C_b) + p_b (1) = 0
    # p_b' = k * (A exp(-k b) dt) * (1 - p_b)
    # Let L_b = A exp(-k b) dt
    # p_b = 1 - exp(-L_b)
    # p_b' = k L_b exp(-L_b)
    
    # We solve for b using Newton's method on g(b) = 0.
    # Initial guess: b0 = max(0, -C_b) (if C_b is negative, we want small b; if positive, larger b)
    # Actually, if C_b > 0, we want to increase b to increase p_b.
    # If C_b < 0, we want to decrease b.
    
    # Boundary handling:
    # If q == Q, no bid fill. b is irrelevant (set to 0).
    # If q == -Q, no ask fill. a is irrelevant (set to 0).
    
    # We will iterate backwards in time.
    # To save time, we can vectorize the Newton steps across states for a given time step.
    
    # Define helper to find optimal depth given C (coefficient) and whether it's bid or ask.
    # For bid: C = D1 - p_a D3
    # For ask: C = D2 - p_b D3
    
    # Since b and a are coupled, we can use an iterative approach:
    # 1. Start with a=0, b=0.
    # 2. Optimize b given a.
    # 3. Optimize a given b.
    # 4. Repeat until convergence (usually 1-2 iterations are enough for smooth functions).
    
    # Let's implement a vectorized Newton solver for the 1D subproblem.
    # We want to find d such that h(d) = p_d' (d + C) + p_d = 0.
    # h(d) = k L_d exp(-L_d) (d + C) + (1 - exp(-L_d))
    # h'(d) = ... complex. Let's just use a small grid + refinement or simple bisection/Newton hybrid.
    # Given the smoothness, a few Newton steps from a good initial guess will work.
    
    # Max depth cap to avoid numerical issues
    d_max = 5.0 / k
    
    def solve_depth(C_vals, p_other_vals, D3_vals):
        """
        Solves for depth d given C = D_other - p_other * D3? No.
        Wait, the equation for b is:
        max_b p_b (b + D1 - p_a D3)
        Let C_b = D1 - p_a D3.
        We need to maximize p_b (b + C_b).
        
        The equation for a is:
        max_a p_a (a + D2 - p_b D3)
        Let C_a = D2 - p_b D3.
        We need to maximize p_a (a + C_a).
        
        This function solves max_d p_d (d + C) for an array of C.
        """
        # C_vals: array of shape (n_states,)
        # We assume p_d depends on d, A, k, dt.
        
        # Initialize d
        # If C is large positive, d should be large.
        # If C is large negative, d should be small (0).
        # Let's use d0 = np.clip(-C_vals, 0, d_max)
        d = np.clip(-C_vals, 0.0, d_max)
        
        # Newton iteration
        for _ in range(5):
            L = A * np.exp(-k * d) * dt
            p = 1.0 - np.exp(-L)
            dp_dd = k * L * np.exp(-L)
            
            # f(d) = p(d) * (d + C)
            # f'(d) = dp_dd * (d + C) + p
            f = p * (d + C_vals)
            fp = dp_dd * (d + C_vals) + p
            
            # Update
            # If fp is close to 0, stop or take small step
            # We want to maximize, so we move in direction of gradient
            # Newton step: d_new = d - f / fp
            # But we are maximizing f(d). If f'(d) > 0, increase d. If f'(d) < 0, decrease d.
            # The root of f'(d) = 0 is the maximum.
            # So we solve fp = 0.
            # d_new = d - fp / fpp? No, we don't have fpp easily.
            # Let's use a simple gradient step with adaptive learning rate?
            # Or just bisection if we know the bounds?
            # Actually, the function p(d)(d+C) is concave for reasonable parameters?
            # Let's check. p(d) is sigmoid-like.
            # For small d, p ~ L ~ exp(-kd).
            # f(d) ~ exp(-kd)(d+C).
            # f'(d) = -k exp(-kd)(d+C) + exp(-kd) = exp(-kd)(1 - k(d+C)).
            # Root at d + C = 1/k => d = 1/k - C.
            # This is a very good initial guess!
            
            d_guess = np.clip(1.0/k - C_vals, 0.0, d_max)
            
            # Let's use a few Newton steps on fp(d) = 0 starting from d_guess
            d = d_guess
            for _ in range(3):
                L = A * np.exp(-k * d) * dt
                p = 1.0 - np.exp(-L)
                dp_dd = k * L * np.exp(-L)
                
                fp = dp_dd * (d + C_vals) + p
                fpp = -k**2 * L * np.exp(-L) * (d + C_vals) + 2 * dp_dd # Approximate 2nd deriv
                # Actually, let's just do a safe update
                # If fp > 0, d is too small. If fp < 0, d is too large.
                # d_new = d - fp / fpp
                # If fpp is small, use gradient descent
                step = fp / np.where(np.abs(fpp) > 1e-6, fpp, 1.0)
                # Limit step size
                step = np.clip(step, -d_max, d_max)
                d_new = d - step
                d_new = np.clip(d_new, 0.0, d_max)
                d = d_new
                
        return d

    # Main loop
    for t in range(n_steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Break and fill remaining with heuristic
            break
            
        V_next = V[t + 1]
        
        # Pre-compute differences for all states
        # V_p = V_next[1:, ]? No, V_next is 1D array of shape (n_states,)
        # V_p[i] = V_next[i+1]
        # V_m[i] = V_next[i-1]
        # V_0[i] = V_next[i]
        
        V_p = np.zeros(n_states)
        V_m = np.zeros(n_states)
        V_0 = V_next.copy()
        
        V_p[1:] = V_next[:-1]  # V_p[i] = V_next[i+1] for i < n_states-1
        V_p[0] = V_next[0]     # Dummy, will be overwritten or ignored at boundary
        V_m[:-1] = V_next[1:]  # V_m[i] = V_next[i-1] for i > 0
        V_m[-1] = V_next[-1]   # Dummy
        
        D1 = V_p - V_0
        D2 = V_m - V_0
        D3 = V_p + V_m - 2 * V_0
        
        # Initialize b and a
        b = np.zeros(n_states)
        a = np.zeros(n_states)
        
        # Iterate to solve coupled system
        # We can do 2 iterations:
        # 1. Assume a=0, solve for b.
        # 2. Use b, solve for a.
        
        # Step 1: Solve for b assuming a=0
        # C_b = D1 - 0 * D3 = D1
        b = solve_depth(D1, None, None)
        
        # Step 2: Solve for a using b
        # p_b is needed
        L_b = A * np.exp(-k * b) * dt
        p_b = 1.0 - np.exp(-L_b)
        
        C_a = D2 - p_b * D3
        a = solve_depth(C_a, None, None)
        
        # Step 3: Refine b using a
        L_a = A * np.exp(-k * a) * dt
        p_a = 1.0 - np.exp(-L_a)
        
        C_b = D1 - p_a * D3
        b = solve_depth(C_b, None, None)
        
        # Step 4: Refine a using b
        L_b = A * np.exp(-k * b) * dt
        p_b = 1.0 - np.exp(-L_b)
        
        C_a = D2 - p_b * D3
        a = solve_depth(C_a, None, None)
        
        # Calculate Value
        L_b = A * np.exp(-k * b) * dt
        L_a = A * np.exp(-k * a) * dt
        p_b = 1.0 - np.exp(-L_b)
        p_a = 1.0 - np.exp(-L_a)
        
        # Cost
        costs = 0.5 * gamma * sigma**2 * q_indices**2 * dt
        
        # Immediate Reward
        reward = p_b * b + p_a * a - costs
        
        # Future Value
        # E[V_next] = V_0 + p_b D1 + p_a D2 - p_b p_a D3
        future = V_0 + p_b * D1 + p_a * D2 - p_b * p_a * D3
        
        V[t+1] = reward + future
        
        # Store actions
        # Boundary conditions:
        # If q == Q, bid is not allowed. Set b=0.
        # If q == -Q, ask is not allowed. Set a=0.
        b[Q + Q] = 0.0  # Index for q=Q is 2Q
        a[0] = 0.0      # Index for q=-Q is 0
        
        opt_bid[t+1] = b
        opt_ask[t+1] = a
        
    _DP_CACHE[key] = (opt_bid, opt_ask)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed DP table.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple heuristic
        k = params['k']
        depth = 1.0 / k
        return np.full(len(inventory), depth), np.full(len(inventory), depth)

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimal actions table
    opt_bid, opt_ask = _solve_dp_continuous(sigma, A, k, gamma, T, steps, Q)
    
    # Map inventory to index
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    # Get actions for step n
    n = int(t / (T / steps))
    n = np.clip(n, 0, steps-1)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Ensure valid depths
    bid_depths = np.clip(bid_depths, 0.0, 5.0 / k)
    ask_depths = np.clip(ask_depths, 0.0, 5.0 / k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
