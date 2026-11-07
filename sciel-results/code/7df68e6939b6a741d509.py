"""Rewritten with continuous HJB-style depth optimization via exact FOC solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _compute_depth(A, k, dt, delta):
    """
    Compute the optimal depth d that maximizes p(d)*(d + delta)
    where p(d) = 1 - exp(-A*exp(-k*d)*dt).
    
    If the maximum value is <= 0, return np.inf (no quote).
    Uses a combination of analytical bounds, bisection, and Newton refinement.
    """
    # The objective f(d) = p(d)*(d+delta) for d >= 0
    # We want to maximize this.
    
    # Quick check: if delta >= 0 and p(0) is small, the optimum might be at 0
    # If delta < 0, we need p(d)*(d+delta) > 0, so d > -delta
    
    # First, check if quoting at all is beneficial
    # f(0) = p(0) * delta where p(0) = 1 - exp(-A*dt)
    p0 = 1.0 - np.exp(-A * dt)
    f0 = p0 * delta
    
    # f(d) as d -> infinity: p(d) -> 1, so f(d) -> d + delta -> infinity
    # Wait, that's not right. As d -> infinity, exp(-k*d) -> 0, so A*exp(-k*d)*dt -> 0
    # So p(d) -> 0. We need to be more careful.
    # p(d) = 1 - exp(-A*exp(-k*d)*dt)
    # For large d: A*exp(-k*d)*dt is small, so p(d) ≈ A*exp(-k*d)*dt
    # f(d) ≈ A*exp(-k*d)*dt * (d + delta)
    # This goes to 0 as d -> infinity.
    
    # So f(d) starts at f(0) = p0*delta, and goes to 0 as d -> infinity.
    # If delta < 0, f(0) < 0, and we need to find if f(d) > 0 for some d.
    # f(d) > 0 requires p(d)*(d+delta) > 0, so d > -delta (if delta < 0).
    
    # The FOC: f'(d) = p'(d)*(d+delta) + p(d) = 0
    # p'(d) = k * A * exp(-k*d) * dt * exp(-A*exp(-k*d)*dt)
    # Let lam = A*exp(-k*d)*dt
    # p(d) = 1 - exp(-lam)
    # p'(d) = k * lam * exp(-lam)
    # FOC: k*lam*exp(-lam)*(d+delta) + (1-exp(-lam)) = 0
    # Let u = lam = A*exp(-k*d)*dt, so d = -(1/k)*ln(u/(A*dt))
    # k*u*exp(-u)*(-(1/k)*ln(u/(A*dt)) + delta) + (1-exp(-u)) = 0
    # -u*exp(-u)*(delta - (1/k)*ln(u/(A*dt))) + (1-exp(-u)) = 0
    
    # Alternative approach: use bisection on the FOC
    # f'(d) = k*lam*exp(-lam)*(d+delta) + (1-exp(-lam))
    # where lam = A*exp(-k*d)*dt
    
    # Let's define g(d) = f'(d) and find where g(d) = 0
    # g(0) = k*A*dt*exp(-A*dt)*delta + (1-exp(-A*dt))
    # For large d, lam -> 0, g(d) -> k*lam*1*(d+delta) + lam = lam*(k*(d+delta) + 1)
    # Since lam -> 0, g(d) -> 0 from positive side if k*(d+delta)+1 > 0
    
    # Actually, let's think about this differently.
    # The optimal d satisfies the FOC if it exists in the interior.
    # Otherwise, d = 0 or d -> infinity (but f -> 0 there).
    
    # Let's use a robust approach:
    # 1. If f(0) <= 0 and delta < 0, check if there's a positive maximum
    # 2. Use bisection to find the FOC root
    # 3. Evaluate f at the root and at 0, pick the better
    
    # For efficiency, let's use an analytical approximation first
    
    # When A*dt is small (which it is: A=140, dt=0.001, so A*dt=0.14),
    # p(d) ≈ A*exp(-k*d)*dt for most d
    # f(d) ≈ A*exp(-k*d)*dt*(d+delta)
    # FOC: A*exp(-k*d)*dt*(1 - k*(d+delta)) = 0
    # So d + delta = 1/k, i.e., d = 1/k - delta
    # But this is only valid when the approximation holds.
    
    # Let's use a hybrid approach:
    # 1. Try the analytical solution d_approx = max(0, 1/k - delta)
    # 2. Refine with Newton/bisection
    # 3. Compare with d=0
    
    # Actually, let's just do a careful bisection + Newton
    
    # Define the derivative g(d) = f'(d)
    def _g(d_val):
        lam = A * np.exp(-k * d_val) * dt
        if lam < 1e-15:
            return 0.0
        exp_neg_lam = np.exp(-lam)
        return k * lam * exp_neg_lam * (d_val + delta) + (1.0 - exp_neg_lam)
    
    # Define the objective f(d)
    def _f(d_val):
        lam = A * np.exp(-k * d_val) * dt
        if lam < 1e-15:
            return 0.0
        return (1.0 - np.exp(-lam)) * (d_val + delta)
    
    # Check if quoting is beneficial at all
    # If delta >= 0, f(0) >= 0, and there might be an interior maximum
    # If delta < 0, we need f(d) > 0 for some d
    
    # Find the range where f(d) > 0
    # f(d) > 0 iff p(d)*(d+delta) > 0
    # Since p(d) > 0 for all finite d, we need d + delta > 0, i.e., d > -delta
    # If delta >= 0, d >= 0 always satisfies this
    # If delta < 0, we need d > -delta
    
    # The maximum of f(d) is either at d=0 (if delta >= 0) or at an interior point
    
    # Let's find the critical point using bisection
    # g(d) = f'(d)
    # g(0) = k*A*dt*exp(-A*dt)*delta + (1-exp(-A*dt))
    # For the FOC to have a solution, we need g to change sign
    
    # Let's evaluate g at some points to find a bracket
    # If g(0) <= 0, the maximum is at d=0
    # If g(0) > 0, there might be an interior maximum where g changes from + to -
    
    g0 = _g(0.0)
    
    if g0 <= 0:
        # Maximum at d=0
        f_at_0 = _f(0.0)
        if f_at_0 <= 0:
            return np.inf
        return 0.0
    
    # g(0) > 0, look for where g becomes negative
    # As d increases, lam decreases, and g(d) should eventually become negative
    # Find a bracket [a, b] where g(a) > 0 and g(b) < 0
    
    a = 0.0
    # Estimate a good upper bound
    # When lam is very small, g(d) ≈ lam*(k*(d+delta)+1)
    # This becomes negative when k*(d+delta)+1 < 0, i.e., d < -delta - 1/k
    # But that doesn't make sense for large d...
    # Actually for large d, lam is tiny, so g(d) ≈ 0 from positive side
    # Let me reconsider: g(d) = k*lam*exp(-lam)*(d+delta) + (1-exp(-lam))
    # For large d, lam ≈ 0, so g(d) ≈ k*lam*(d+delta) + lam = lam*(k*(d+delta)+1)
    # If d+delta > 0 and k*(d+delta)+1 > 0, then g(d) > 0 for large d
    # This means f is still increasing, which contradicts f -> 0
    # The issue is that for very large d, lam is so small that the approximation breaks
    
    # Let's just search for a bracket
    b = a
    for _ in range(100):
        b = b * 2 + 0.1
        if b > 100.0:
            break
        gb = _g(b)
        if gb < 0:
            break
    else:
        # Couldn't find a sign change; the maximum might be at a very large d
        # or the function is monotonically increasing (unlikely given f -> 0)
        # Just return a large depth
        return min(100.0, max(0.0, 1.0/k - delta))
    
    # Bisection to find where g(d) = 0
    for _ in range(60):
        mid = 0.5 * (a + b)
        gm = _g(mid)
        if abs(gm) < 1e-15:
            a = b = mid
            break
        if gm > 0:
            a = mid
        else:
            b = mid
    
    d_star = 0.5 * (a + b)
    
    # Evaluate f at d_star and at 0
    f_star = _f(d_star)
    f_zero = _f(0.0)
    
    if f_star > f_zero and f_star > 0:
        return d_star
    elif f_zero > 0:
        return 0.0
    else:
        return np.inf


def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Value table
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        V_next = V[n + 1]
        
        # Delta V for bid (inventory increases by 1)
        delta_b = np.zeros(2 * Q + 1)
        if Q > 0:
            delta_b[:-1] = V_next[1:] - V_next[:-1]
        
        # Delta V for ask (inventory decreases by 1)
        delta_a = np.zeros(2 * Q + 1)
        if Q > 0:
            delta_a[1:] = V_next[:-1] - V_next[1:]
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        ask_depths = np.full(2 * Q + 1, np.inf)
        bid_rewards = np.zeros(2 * Q + 1)
        ask_rewards = np.zeros(2 * Q + 1)
        
        # Compute optimal depths
        for q in range(2 * Q + 1):
            # Bid side
            if q < 2 * Q:  # Not at max inventory
                d_b = _compute_depth(A, k, dt, delta_b[q])
                if np.isfinite(d_b):
                    bid_depths[q] = d_b
                    lam = A * np.exp(-k * d_b) * dt
                    if lam < 1e-15:
                        p_b = lam
                    else:
                        p_b = 1.0 - np.exp(-lam)
                    bid_rewards[q] = max(p_b * (d_b + delta_b[q]), 0.0)
            
            # Ask side
            if q > 0:  # Not at min inventory
                d_a = _compute_depth(A, k, dt, delta_a[q])
                if np.isfinite(d_a):
                    ask_depths[q] = d_a
                    lam = A * np.exp(-k * d_a) * dt
                    if lam < 1e-15:
                        p_a = lam
                    else:
                        p_a = 1.0 - np.exp(-lam)
                    ask_rewards[q] = max(p_a * (d_a + delta_a[q]), 0.0)
        
        # Update Value
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        n_paths = len(inventory)
        return np.full(n_paths, np.inf), np.full(n_paths, np.inf)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
