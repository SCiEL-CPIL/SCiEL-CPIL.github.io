"""
Optimal market making via DP with continuous Newton refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _heuristic_depths(params, t, inventory):
    """Analytical fallback: time-varying base depth with inventory skew."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0 / k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0 / k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _solve_foc(d_other, delta_plus, delta_minus, delta_2, c_inv, A, k, dt):
    """
    Solve for depth d such that the FOC of the Bellman equation is satisfied,
    given the other depth d_other.
    Equation: 1 - p(d) = A * exp(-k*d) * (d*delta_plus + d_other*delta_minus + d*d_other*delta_2 - c_inv) / d
    Rearranged: f(d) = 1 - (1 - exp(-A exp(-k d) dt)) - A exp(-k d) (delta_plus + (d_other*delta_minus + d_other^2*delta_2 - c_inv)/d)
    Note: The term inside the parenthesis in the FOC derivation:
    d * Delta_plus + d_other * Delta_minus + d * d_other * Delta_2 - c_inv
    Divided by d: Delta_plus + (d_other * Delta_minus + d * d_other * Delta_2 - c_inv) / d
    = Delta_plus + d_other * Delta_minus / d + d_other * Delta_2 - c_inv / d
    = (Delta_plus + d_other * Delta_2) + (d_other * Delta_minus - c_inv) / d
    
    So f(d) = exp(-A exp(-k d) dt) - A exp(-k d) * [ (Delta_plus + d_other*Delta_2) + (d_other*Delta_minus - c_inv)/d ]
    We want f(d) = 0.
    """
    if d_other < 1e-6:
        # If other depth is effectively 0 (no quote), the problem decouples or is trivial.
        # In our context, d_other > 0 always unless boundary.
        pass

    # Constants for the equation
    C1 = delta_plus + d_other * delta_2
    C2 = d_other * delta_minus - c_inv
    
    # We solve for d in log space to ensure positivity and stability
    # Let x = log(d), so d = exp(x)
    # f(x) = exp(-A exp(-k exp(x)) dt) - A exp(-k exp(x)) * (C1 + C2 * exp(-x))
    
    # Newton on x
    x = np.log(max(0.01 / k, 1e-6)) # Initial guess
    
    for _ in range(20):
        d = np.exp(x)
        exp_kd = np.exp(-k * d)
        p = 1.0 - np.exp(-A * exp_kd * dt)
        
        # f(x) = p - A * exp_kd * (C1 + C2/d)
        # Note: 1 - p = A exp_kd (C1 + C2/d) is the target equality
        # So f(x) = (1 - p) - A * exp_kd * (C1 + C2/d)
        # We want f(x) = 0
        
        term1 = 1.0 - p
        term2 = A * exp_kd * (C1 + C2 / d)
        f_val = term1 - term2
        
        if abs(f_val) < 1e-10:
            break
            
        # Derivative f'(x)
        # d/dx (1 - p) = d/dx (1 - exp(-A e^{-kd} dt)) = -exp(-A e^{-kd} dt) * (-A e^{-kd} * (-k) * d * dx/dd? No, d/dd * dd/dx)
        # dp/dd = -exp(-A e^{-kd} dt) * (-A e^{-kd}) * (-k) = -A k e^{-kd} exp(-A e^{-kd} dt)
        # dd/dx = d
        # d(1-p)/dx = -dp/dx = -dp/dd * d = A k e^{-kd} exp(-A e^{-kd} dt) * d
        
        # d/dx (A e^{-kd} (C1 + C2/d))
        # Let g(d) = A e^{-kd} (C1 + C2/d)
        # g'(d) = A [ -k e^{-kd} (C1 + C2/d) + e^{-kd} (-C2/d^2) ]
        # g'(x) = g'(d) * d = A [ -k d e^{-kd} (C1 + C2/d) - C2 e^{-kd} / d ]
        #        = A e^{-kd} [ -k d C1 - k C2 - C2/d ]
        
        exp_p_term = np.exp(-A * exp_kd * dt)
        df_d_1 = A * k * d * exp_kd * exp_p_term
        df_d_2 = A * exp_kd * (-k * d * C1 - k * C2 - C2 / d)
        
        f_prime = df_d_1 - df_d_2
        
        if abs(f_prime) < 1e-12:
            break
            
        step = f_val / f_prime
        # Limit step size to prevent divergence
        step = np.clip(step, -2.0, 2.0)
        x -= step
        
        # Keep d in reasonable bounds
        d_min = 0.001
        d_max = 50.0 / k
        if np.exp(x) < d_min:
            x = np.log(d_min)
        elif np.exp(x) > d_max:
            x = np.log(d_max)
            
    d_opt = np.exp(x)
    return np.clip(d_opt, 0.001, 50.0 / k)


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP
    with continuous Newton refinement.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt

    # Initial guess for depths: 1/k
    d_init = 1.0 / k

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            # Fallback: fill remaining with heuristic-like values
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    # Simple linear skew
                    d = d_init + 0.1 * q * (T - n2*dt)
                    d = max(d, 0.01/k)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            bid_table[:, n_q - 1] = np.inf
            ask_table[:, 0] = np.inf
            return bid_table, ask_table

        Vn1 = V[n + 1]

        # Precompute V differences
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1

        for qi in range(n_q):
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val

            if qi == n_q - 1:  # q = Q, no bid
                bid_table[n, qi] = np.inf
                # Solve for ask only? If bid is inf, p_b=0.
                # J(d_a) = d_a p_a + Vn1 + p_a Dm1 - cost
                # Maximize d_a p_a + p_a Dm1 = p_a (d_a + Dm1)
                # FOC for d_a: 1 - p_a = A e^{-k d_a} (1) / (d_a + Dm1)? No.
                # If p_b=0, the cross term vanishes.
                # Maximize (1 - exp(-A e^{-k d_a} dt)) (d_a + Dm1)
                # This is a 1D problem. We can use a small grid or Newton.
                # Let's use a small grid for robustness in boundary cases.
                grid = np.geomspace(0.01/k, 10/k, 50)
                pa = 1.0 - np.exp(-A * np.exp(-k * grid) * dt)
                vals = pa * (grid + Dm1[qi]) - cost
                best_idx = np.argmax(vals)
                ask_table[n, qi] = grid[best_idx]
                V[n, qi] = vals[best_idx]
                continue

            if qi == 0:  # q = -Q, no ask
                ask_table[n, qi] = np.inf
                # Solve for bid only
                grid = np.geomspace(0.01/k, 10/k, 50)
                pb = 1.0 - np.exp(-A * np.exp(-k * grid) * dt)
                vals = pb * (grid + D1[qi]) - cost
                best_idx = np.argmax(vals)
                bid_table[n, qi] = grid[best_idx]
                V[n, qi] = vals[best_idx]
                continue

            # Interior state: Solve coupled system
            # Start with previous step's values or heuristic
            if n < steps - 1:
                d_b = bid_table[n+1, qi]
                d_a = ask_table[n+1, qi]
            else:
                d_b = d_init
                d_a = d_init
            
            # Ensure they are valid starting points
            if not np.isfinite(d_b) or d_b < 0.01/k: d_b = d_init
            if not np.isfinite(d_a) or d_a < 0.01/k: d_a = d_init

            d_b_curr = d_b
            d_a_curr = d_a
            
            for _ in range(10):
                # Solve for d_b given d_a_curr
                d_b_new = _solve_foc(d_a_curr, D1[qi], Dm1[qi], D2[qi], cost, A, k, dt)
                # Solve for d_a given d_b_new
                d_a_new = _solve_foc(d_b_new, Dm1[qi], D1[qi], D2[qi], cost, A, k, dt)
                
                if abs(d_b_new - d_b_curr) < 1e-5 and abs(d_a_new - d_a_curr) < 1e-5:
                    d_b_curr = d_b_new
                    d_a_curr = d_a_new
                    break
                d_b_curr = d_b_new
                d_a_curr = d_a_new
                
            bid_table[n, qi] = d_b_curr
            ask_table[n, qi] = d_a_curr
            
            # Calculate Value
            p_b = 1.0 - np.exp(-A * np.exp(-k * d_b_curr) * dt)
            p_a = 1.0 - np.exp(-A * np.exp(-k * d_a_curr) * dt)
            
            # J = d_b p_b + d_a p_a + Vn1 + p_b D1 + p_a Dm - p_b p_a D2 - cost
            V[n, qi] = d_b_curr * p_b + d_a_curr * p_a + Vn1[qi] + p_b * D1[qi] + p_a * Dm1[qi] - p_b * p_a * D2[qi] - cost

    # Post-process boundaries
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    # Ensure valid values
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
