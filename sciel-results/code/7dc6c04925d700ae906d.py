import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 280.0
_CACHE = {}

def _build_phi_inverse(A, k, dt):
    """
    Precompute the inverse of the optimality condition function.
    We want to solve for x in:
    x * (1 - exp(-lambda * x)) + C * x = max
    
    Let f(x) = x(1 - exp(-lambda x)) + C x.
    f'(x) = (1 - exp(-lambda x)) + x(lambda exp(-lambda x)) + C
    f'(x) = 1 + C + lambda x exp(-lambda x) - exp(-lambda x)
    f'(x) = 1 + C + exp(-lambda x) (lambda x - 1)
    
    We define phi(x) = f'(x) - 1 = C + exp(-lambda x)(lambda x - 1).
    We want phi(x) = C.
    Since phi(x) is monotonic (decreasing then increasing? No, let's check).
    Actually, simpler:
    The term to maximize is g(x) = x(1 - e^{-\lambda x}) + C x.
    g'(x) = 1 - e^{-\lambda x} + \lambda x e^{-\lambda x} + C.
    Set g'(x) = 0.
    1 + C = e^{-\lambda x} (1 - \lambda x).
    Let H(x) = e^{-\lambda x} (1 - \lambda x).
    We need to solve H(x) = 1 + C.
    
    Note: If C is large positive, 1+C > 1. But max(H(x)) is at x=0, H(0)=1.
    Wait, if C > 0, we are adding profit from inventory change.
    Let's re-verify the objective.
    Objective: P(d) * d + P(d) * Delta.
    P(d) = 1 - exp(-A exp(-kd) dt).
    Let x = exp(-kd). P = 1 - exp(-A dt x).
    d = -1/k ln x.
    Term = (1 - exp(-A dt x)) * (-1/k ln x) + (1 - exp(-A dt x)) * Delta.
    Let F(x) = (1 - exp(-\lambda x)) (-1/k ln x + Delta).
    We want to maximize F(x) for x in (0, 1].
    
    Derivative F'(x):
    Let u = 1 - exp(-\lambda x). u' = \lambda exp(-\lambda x).
    Let v = -1/k ln x + Delta. v' = -1/(k x).
    F'(x) = u' v + u v'
    = \lambda e^{-\lambda x} (-1/k ln x + Delta) + (1 - e^{-\lambda x}) (-1/(k x))
    
    Set F'(x) = 0:
    \lambda e^{-\lambda x} (-1/k ln x + Delta) = (1 - e^{-\lambda x}) / (k x)
    k \lambda e^{-\lambda x} (-1/k ln x + Delta) = (1 - e^{-\lambda x}) / x
    -\lambda e^{-\lambda x} \ln x + k \lambda \Delta e^{-\lambda x} = (1 - e^{-\lambda x}) / x
    
    This looks messy to invert analytically.
    However, we can just build a lookup table for the optimal x given C (where C is effectively the Delta term scaled).
    Actually, the structure is:
    Maximize h(x) = (1 - e^{-\lambda x}) (-1/k ln x) + C (1 - e^{-\lambda x})
    = (1 - e^{-\lambda x}) (C - 1/k ln x)
    
    Let's just use the derivative condition on a grid of C values.
    For a given C, find x that maximizes.
    Since we need to do this for many C values (Delta), we can precompute a table mapping C -> x_opt.
    C range: Delta is V[q+1]-V[q]. V is bounded.
    """
    lambda_t = A * dt
    
    # Grid for x (hazard rate)
    # x in (0, 1].
    # As x -> 0, P -> 0, d -> inf.
    # As x -> 1, P -> 1-exp(-lambda), d -> 0.
    
    # We need to cover a range of C.
    # Let's determine a safe range for C.
    # Max possible gain from one step is roughly A * dt * max_depth.
    # But Delta comes from V.
    # Let's assume C is in [-10, 10] for safety, or wider.
    
    # Instead of inverting derivative, let's just evaluate the objective on a fine grid of x
    # for a set of C values, and store the argmax.
    
    # Grid for x
    x_grid = np.logspace(-10, 0, 2000)
    
    # Precompute P(x) and d(x)
    P_x = 1.0 - np.exp(-lambda_t * x_grid)
    d_x = -np.log(x_grid) / k
    
    # Objective matrix: Obj[C_idx, x_idx]
    # We need to handle a range of C.
    # Let's pick a range for C based on typical V differences.
    # V is sum of gains. Max gain ~ A*dt*max_d.
    # Let's say C in [-5, 5] is enough. If not, we clip.
    # Actually, let's make it dynamic or wider.
    # Let's use C_grid from -10 to 10 with 2000 points.
    C_grid = np.linspace(-10, 10, 2000)
    
    # Obj[i, j] = P_x[j] * (d_x[j] + C_grid[i])
    # Note: The term is P(x) * d + P(x) * Delta = P(x) * (d + Delta).
    # So C corresponds to Delta.
    
    # Vectorized computation
    # P_x is (2000,), d_x is (2000,), C_grid is (2000,)
    # We want matrix M (2000, 2000) where M[i, j] = P_x[j] * (d_x[j] + C_grid[i])
    
    # To save memory/time, we can compute this in chunks or just do it once.
    # 2000x2000 is 4M floats, ~32MB. Acceptable.
    
    M = P_x[None, :] * (d_x[None, :] + C_grid[:, None])
    
    # Find argmax over x (axis 1) for each C
    best_x_idx = np.argmax(M, axis=1)
    
    # We also need to know if the max value is <= 0.
    # If max value <= 0, we should not quote (x=0, d=inf).
    max_vals = np.max(M, axis=1)
    
    # Store the results
    # We want to interpolate: given a C, find x_opt.
    # C_grid is linear. x_opt is not necessarily smooth but monotonic?
    # Let's check monotonicity.
    # If C increases (more profit from selling), we want to quote tighter (smaller d, larger x).
    # So x_opt should increase with C.
    
    return C_grid, x_grid[best_x_idx], max_vals

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Precompute the C -> x_opt mapping
    C_grid, x_opt_grid, max_val_grid = _build_phi_inverse(A, k, dt)
    
    # For interpolation, we need x_opt as a function of C.
    # np.interp requires xp to be increasing. C_grid is increasing.
    # x_opt_grid should be roughly increasing.
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 20:
            # Fallback: use a default depth
            default_depth = 1.0 / k
            for tt in range(t, N):
                opt_b[tt, :] = default_depth
                opt_a[tt, :] = default_depth
            return opt_b, opt_a
        
        V_next = V[t + 1, :]
        
        # Deltas
        delta_plus = np.zeros(n_q)
        delta_minus = np.zeros(n_q)
        
        if n_q > 2:
            delta_plus[1:n_q-1] = V_next[2:n_q] - V_next[1:n_q-1]
            delta_minus[1:n_q-1] = V_next[0:n_q-2] - V_next[1:n_q-1]
            # Boundaries
            delta_plus[n_q-1] = 0.0 # No bid at Q
            delta_minus[0] = 0.0     # No ask at -Q
            
        # Fixed point iteration for bid and ask
        # Initialize
        b_depths = np.full(n_q, 1.0/k)
        a_depths = np.full(n_q, 1.0/k)
        
        # We iterate a few times to converge b and a
        for _iter in range(10):
            # Calculate probabilities from current depths
            # P_b = 1 - exp(-A exp(-k b) dt)
            # P_a = 1 - exp(-A exp(-k a) dt)
            
            # Handle inf depths
            # If depth is inf, P is 0.
            # Our lookup table handles x in (0,1].
            # If we decide not to quote, we set x=0 (P=0).
            
            # Calculate effective C for bid
            # C_b = (1 - P_a) * delta_plus - P_a * delta_minus
            # Wait, the objective is:
            # E[Gain] = P_b * (b + delta_plus) * (1 - P_a) + P_b * (b - delta_minus) * P_a + (1-P_b)*...
            # Actually, the transition is:
            # If bid hit: q -> q+1, gain b. Prob P_b.
            # If ask hit: q -> q-1, gain a. Prob P_a.
            # If both: q -> q, gain b+a. Prob P_b P_a.
            # If none: q -> q, gain 0. Prob (1-P_b)(1-P_a).
            
            # Expected Value Change from fills:
            # Delta V = P_b(1-P_a) (V[q+1] - V[q] + b) + P_a(1-P_b) (V[q-1] - V[q] + a) + P_b P_a (V[q] - V[q] + b + a)
            # Note: V[q+1]-V[q] is delta_plus. V[q-1]-V[q] is delta_minus.
            
            # Term for b:
            # b is earned if bid hits.
            # Prob bid hits = P_b.
            # So b * P_b.
            # Impact on V:
            # P_b (1-P_a) delta_plus + P_b P_a * 0 (V change is 0 for q->q)
            # Wait, if both hit, q stays q. V change is 0.
            # So V impact from b decision depends on P_a.
            # V_impact_b = P_b [ (1-P_a) delta_plus + P_a * 0 ] = P_b (1-P_a) delta_plus
            
            # Similarly for a:
            # V_impact_a = P_a (1-P_b) delta_minus
            
            # Total Objective to maximize for b (given a):
            # J_b = P_b * b + P_b (1-P_a) delta_plus
            # Let P_b = 1 - exp(-A e^{-kb} dt).
            # Let x_b = e^{-kb}. P_b = 1 - exp(-A dt x_b).
            # J_b(x_b) = (1 - exp(-A dt x_b)) * (-1/k ln x_b) + (1 - exp(-A dt x_b)) * (1-P_a) delta_plus
            # Let C_b = (1-P_a) delta_plus.
            # J_b(x_b) = (1 - exp(-A dt x_b)) * ( -1/k ln x_b + C_b )
            
            # This matches our precomputed table!
            # We need C_b for each q.
            
            P_a_curr = 1.0 - np.exp(-A * np.exp(-k * a_depths) * dt)
            # Clip P_a to [0,1]
            
            C_b = (1.0 - P_a_curr) * delta_plus
            
            # Interpolate x_opt for each C_b
            # C_b might be outside C_grid range. Clip it.
            C_b_clipped = np.clip(C_b, C_grid[0], C_grid[-1])
            
            x_b_opt = np.interp(C_b_clipped, C_grid, x_opt_grid)
            
            # Check if we should quote at all.
            # If max_val for this C is <= 0, we should set x=0 (P=0, d=inf).
            # We need to interpolate max_val too? Or just check sign.
            # max_val_grid is the max value of J for that C.
            # If max_val <= 0, don't quote.
            max_val_b = np.interp(C_b_clipped, C_grid, max_val_grid)
            
            # If max_val_b <= 0, set x=0.
            x_b_opt[max_val_b <= 1e-9] = 0.0
            
            # Convert x to depth
            # d = -1/k ln x. If x=0, d=inf.
            # np.log(0) is -inf. -1/k * -inf = inf.
            # But np.log(0) emits warning. Use where.
            new_b_depths = np.where(x_b_opt > 1e-12, -np.log(x_b_opt) / k, np.inf)
            
            # Update b_depths
            b_depths = new_b_depths
            
            # Now for a
            P_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_depths) * dt)
            
            C_a = (1.0 - P_b_curr) * delta_minus
            
            C_a_clipped = np.clip(C_a, C_grid[0], C_grid[-1])
            x_a_opt = np.interp(C_a_clipped, C_grid, x_opt_grid)
            
            max_val_a = np.interp(C_a_clipped, C_grid, max_val_grid)
            x_a_opt[max_val_a <= 1e-9] = 0.0
            
            new_a_depths = np.where(x_a_opt > 1e-12, -np.log(x_a_opt) / k, np.inf)
            a_depths = new_a_depths
            
            # Check convergence? Or just fixed 10 iterations.
            # 10 iterations is fast and usually converges well.
            
        # Force boundaries
        # q = Q (idx n_q-1): No bid.
        b_depths[n_q-1] = np.inf
        # q = -Q (idx 0): No ask.
        a_depths[0] = np.inf
        
        opt_b[t, :] = b_depths
        opt_a[t, :] = a_depths
        
        # Update V[t]
        # V[t, q] = V[t+1, q] + Cost + Expected Gain
        # Expected Gain = P_b b + P_a a + P_b(1-P_a) delta_plus + P_a(1-P_b) delta_minus + P_b P_a (b+a)
        # Wait, let's re-calculate Expected Gain carefully.
        # Gain = (bid_hit ? b : 0) + (ask_hit ? a : 0)
        # V_change = (bid_hit ? delta_plus : 0) + (ask_hit ? delta_minus : 0)
        # Note: if both hit, q doesn't change, so V change is 0.
        # My previous delta_plus/minus definitions:
        # delta_plus = V[q+1] - V[q]
        # delta_minus = V[q-1] - V[q]
        
        # E[Gain + V_change]
        # = P_b(1-P_a) (b + delta_plus) + P_a(1-P_b) (a + delta_minus) + P_b P_a (b + a + 0)
        
        # We have b_depths, a_depths.
        # Calculate P_b, P_a
        P_b = 1.0 - np.exp(-A * np.exp(-k * b_depths) * dt)
        P_a = 1.0 - np.exp(-A * np.exp(-k * a_depths) * dt)
        
        # Handle inf depths -> P=0
        P_b[b_depths == np.inf] = 0.0
        P_a[a_depths == np.inf] = 0.0
        
        # b and a values for gain
        # If depth is inf, gain is 0 (since P=0, term vanishes).
        # But we need b value for the formula.
        # b_val = where(isinf, 0, b_depths)
        b_val = np.where(b_depths == np.inf, 0.0, b_depths)
        a_val = np.where(a_depths == np.inf, 0.0, a_depths)
        
        # Terms
        term1 = P_b * (1.0 - P_a) * (b_val + delta_plus)
        term2 = P_a * (1.0 - P_b) * (a_val + delta_minus)
        term3 = P_b * P_a * (b_val + a_val)
        
        expected_gain = term1 + term2 + term3
        
        # Cost
        q_vals = np.arange(-Q, Q + 1)
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V[t, :] = V_next + cost + expected_gain
        
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _CACHE:
        _CACHE[key] = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    
    opt_b, opt_a = _CACHE[key]
    
    t_idx = int(t / (T / steps))
    if t_idx >= steps:
        t_idx = steps - 1
        
    q_idx = inventory + Q
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    return b_depths, a_depths
