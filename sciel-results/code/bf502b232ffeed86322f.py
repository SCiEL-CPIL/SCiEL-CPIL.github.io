"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies per regime
_policy_cache = {}


def _solve_depth_analytical(delta, A, k, dt):
    """
    Solve for optimal depth d maximizing h(d) = p(d)*(d + delta)
    where p(d) = 1 - exp(-A*exp(-k*d)*dt)
    
    Using the substitution u = exp(-k*d), we get:
    h(u) = (1 - exp(-A*u*dt)) * (-ln(u)/k + delta)
    
    The first-order condition gives a 1D root-finding problem in u.
    We use Newton's method with careful initialization.
    
    Returns: optimal depth d >= 0, or np.inf if no quote is optimal.
    """
    # If delta is very negative, no quote is optimal (h(0) = 0, h goes negative)
    if delta < -1e-10:
        # Check if h(0+) > 0: h'(0+) = p'(0)*(delta) + p(0) = -k*A*dt*delta + 0
        # If delta < 0, h'(0+) > 0 only if... wait, h(d) ≈ A*exp(-k*d)*dt*(d+delta) for small d
        # h(0) = 0 (since p(0)=0 when d=0 exactly... actually p(0) = 1-exp(-A*dt) which may be > 0)
        # Let me reconsider. At d=0: p = 1-exp(-A*dt), h(0) = (1-exp(-A*dt))*delta
        # If delta < 0, h(0) < 0, so we should check if there's a positive d where h > 0
        # For large d, p(d) -> 1, h(d) -> d + delta -> infinity. So there's always an optimum.
        # Actually wait - if delta is very negative, the optimum might be at d=0 with h(0) < 0
        # but we can always choose d=0 and get h(0). If h(0) < 0, we'd prefer no quote.
        # But the problem says we can quote at any depth >= 0. If all h(d) < 0, we quote at inf.
        
        # Check h at d=0:
        p0 = 1.0 - np.exp(-A * dt)
        h0 = p0 * delta
        if h0 <= 0:
            return np.inf
        
        # Check if there's a better d > 0
        # For large d, h(d) ≈ d + delta which is positive for d > -delta
        # So there must be a maximum. Let's find it numerically.
        pass
    
    # Use Newton's method on the first-order condition
    # h'(d) = p'(d)*(d+delta) + p(d) = 0
    # p'(d) = -k * A * exp(-k*d) * dt * (1 - p(d))
    # Let lam = A*exp(-k*d)*dt
    # p = 1 - exp(-lam)
    # p' = -k * lam * exp(-lam)
    
    # FOC: -k*lam*exp(-lam)*(d+delta) + (1-exp(-lam)) = 0
    # Let u = exp(-k*d), so d = -ln(u)/k
    # lam = A*u*dt
    # FOC: -k*(A*u*dt)*exp(-A*u*dt)*(-ln(u)/k + delta) + (1-exp(-A*u*dt)) = 0
    # => (A*u*dt)*exp(-A*u*dt)*(ln(u)/k - delta) + (1-exp(-A*u*dt)) = 0
    # => (A*u*dt)*exp(-A*u*dt)*(ln(u)/k - delta) = exp(-A*u*dt) - 1
    # => (A*u*dt)*(ln(u)/k - delta) = 1 - exp(A*u*dt)
    
    # Let's define f(u) = (A*u*dt)*(ln(u)/k - delta) - (1 - exp(A*u*dt))
    # We want f(u) = 0 for u in (0, 1]
    
    # At u=1 (d=0): f(1) = A*dt*(0 - delta) - (1 - exp(A*dt)) = -A*dt*delta - 1 + exp(A*dt)
    # At u->0: f(u) -> 0*(ln(u)/k - delta) - (1-1) = 0... need to be more careful
    # As u->0: A*u*dt*ln(u)/k -> 0 (since u*ln(u) -> 0), A*u*dt*(-delta) -> 0, 1-exp(A*u*dt) -> A*u*dt
    # So f(u) -> 0 - A*u*dt -> 0 from below
    
    # This is getting complex. Let me just use a robust numerical approach.
    
    # Define the objective and its derivative in terms of d
    def h(d):
        if d < 0:
            return -np.inf
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        return p * (d + delta)
    
    def h_prime(d):
        if d < 0:
            return 0
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        p_prime = -k * lam * np.exp(-lam)
        return p_prime * (d + delta) + p
    
    # Check boundary: if h'(0) <= 0, optimum is at d=0
    hp0 = h_prime(0.0)
    if hp0 <= 0:
        # Optimum at d=0
        h0 = h(0.0)
        if h0 <= 0:
            return np.inf
        return 0.0
    
    # If h'(0) > 0, there's an interior optimum
    # Use Newton's method starting from a reasonable initial guess
    # For small lambda, the optimum is approximately where p*(d+delta) is maximized
    # Approximate: if A*dt is small, p ≈ A*exp(-k*d)*dt, so h ≈ A*dt*exp(-k*d)*(d+delta)
    # Maximize: d/d_d [exp(-k*d)*(d+delta)] = 0 => -k*exp(-k*d)*(d+delta) + exp(-k*d) = 0
    # => k*(d+delta) = 1 => d = 1/k - delta
    
    d_init = max(0.0, 1.0/k - delta)
    d_init = min(d_init, 50.0)  # Cap to avoid issues
    
    # Newton's method
    d = d_init
    for _ in range(50):
        hp = h_prime(d)
        if abs(hp) < 1e-12:
            break
        # Second derivative for Newton (optional, but let's use secant if needed)
        # h''(d) = p''(d)*(d+delta) + 2*p'(d)
        # p''(d) = k^2*lam*exp(-lam) - k^2*lam^2*exp(-lam) = k^2*lam*exp(-lam)*(1-lam)
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        p_prime = -k * lam * np.exp(-lam)
        p_double_prime = k**2 * lam * np.exp(-lam) * (1.0 - lam)
        hp_prime = p_double_prime * (d + delta) + 2 * p_prime
        
        if abs(hp_prime) < 1e-15:
            # Use secant method instead
            d_new = d - hp / 1e-8  # Small step
        else:
            d_new = d - hp / hp_prime
        
        # Ensure d_new >= 0
        d_new = max(0.0, d_new)
        
        if abs(d_new - d) < 1e-10:
            d = d_new
            break
        d = d_new
    
    d = max(0.0, d)
    
    # Verify this is a maximum
    hd = h(d)
    if hd <= 0:
        return np.inf
    
    return d


def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via vectorized dynamic programming
    with analytical depth optimization.
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
    q_values = np.arange(-Q, Q + 1)

    # Value function
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Precompute cost
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        # Compute delta values
        delta_b = np.zeros(n_states - 1)
        delta_b[:] = V_next[1:] - V_next[:-1]
        
        delta_a = np.zeros(n_states - 1)
        delta_a[:] = V_next[:-1] - V_next[1:]
        
        # Solve for optimal depths analytically
        # Vectorize: solve for each delta
        bid_depths = np.array([_solve_depth_analytical(dv, A, k, dt) for dv in delta_b])
        ask_depths = np.array([_solve_depth_analytical(dv, A, k, dt) for dv in delta_a])
        
        opt_bid[n, :-1] = bid_depths
        opt_ask[n, 1:] = ask_depths
        
        # Boundary: no bid at q=Q, no ask at q=-Q
        opt_bid[n, -1] = np.inf
        opt_ask[n, 0] = np.inf
        
        # Compute fill probabilities
        p_b = np.zeros(n_states)
        p_a = np.zeros(n_states)
        
        # For states 0 to n_states-2
        finite_bid = np.isfinite(opt_bid[n, :-1])
        if np.any(finite_bid):
            lam_b = A * np.exp(-k * opt_bid[n, :-1][finite_bid]) * dt
            p_b[:-1][finite_bid] = 1.0 - np.exp(-lam_b)
        
        # For states 1 to n_states-1
        finite_ask = np.isfinite(opt_ask[n, 1:])
        if np.any(finite_ask):
            lam_a = A * np.exp(-k * opt_ask[n, 1:][finite_ask]) * dt
            p_a[1:][finite_ask] = 1.0 - np.exp(-lam_a)
        
        # Compute expected value
        V_next_shift_up = np.zeros(n_states)
        V_next_shift_up[:-1] = V_next[1:]
        
        V_next_shift_down = np.zeros(n_states)
        V_next_shift_down[1:] = V_next[:-1]
        
        # Earnings
        earn_b = p_b * opt_bid[n]
        earn_a = p_a * opt_ask[n]
        earn_b[~np.isfinite(earn_b)] = 0.0
        earn_a[~np.isfinite(earn_a)] = 0.0
        
        # Probabilities
        prob_b_only = p_b * (1.0 - p_a)
        prob_a_only = (1.0 - p_b) * p_a
        prob_none = 1.0 - p_b - p_a + p_b * p_a
        
        v_next_expected = prob_b_only * V_next_shift_up + prob_a_only * V_next_shift_down + prob_none * V_next
        
        V_curr = earn_b + earn_a + v_next_expected - cost
        
        V_next = V_curr

    return opt_bid, opt_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    start_time = time.time()
    
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute policy
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()
    
    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    idx = inventory + Q
    idx = np.clip(idx, 0, 2*Q)
    
    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Ensure valid outputs
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
