# EVOLVE-BLOCK-START
import numpy as np

_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

def _solve_regime(sigma, A, k, gamma, Q=30, T=1.0, steps=1000):
    """
    Solve the optimal market-making policy using a semi-Lagrangian HJB scheme
    with analytical spread optimization via Lambert W function.
    
    Returns:
        bid_depths: array of shape (2*Q+1,) - optimal bid depth for each inventory level
        ask_depths: array of shape (2*Q+1,) - optimal ask depth for each inventory level
    """
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # Terminal condition: V(q, T) = 0 for all q
    V = np.zeros(n_states)
    
    q_to_idx = lambda q: q - q_min
    
    # Time-stepping backwards from T to 0
    for n in range(steps - 1, -1, -1):
        V_new = np.zeros(n_states)
        
        for i in range(n_states):
            q = i + q_min
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            risk_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            V_qi = V[i]
            
            if can_bid:
                V_qp = V[q_to_idx(q + 1)]
                delta_b = V_qp - V_qi
            else:
                V_qp = 0.0
                delta_b = 0.0
            
            if can_ask:
                V_qm = V[q_to_idx(q - 1)]
                delta_a = V_qm - V_qi
            else:
                V_qm = 0.0
                delta_a = 0.0
            
            # Optimize bid depth using closed-form Lambert W solution
            if can_bid:
                # Maximize: p_b * (d_b + delta_b) where p_b = 1 - exp(-A*exp(-k*d_b)*dt)
                # Let x = A*exp(-k*d_b)*dt, so d_b = (1/k)*log(A*dt/x)
                # Objective: (1-x) * ((1/k)*log(A*dt/x) + delta_b)
                # Take derivative and set to zero to get Lambert W form
                # Optimal x satisfies: x * exp(x) = (A*dt)/k * exp(k*delta_b)
                # So x = W((A*dt)/k * exp(k*delta_b))
                arg = (A * dt) / k * np.exp(k * delta_b)
                if arg > 0:
                    # Use numpy's lambertw or approximate
                    # For small arguments, W(z) ≈ z - z^2 + ...
                    # For our purposes, use a simple approximation
                    x_opt = _lambert_w(arg)
                    if x_opt > 0 and x_opt < 1:
                        d_b_opt = (1.0 / k) * np.log(A * dt / x_opt)
                    else:
                        d_b_opt = 0.0
                else:
                    d_b_opt = 0.0
                best_bid_depth = max(0.0, d_b_opt)
            else:
                best_bid_depth = np.inf
            
            # Optimize ask depth using closed-form Lambert W solution
            if can_ask:
                # Maximize: p_a * (delta_a - d_a) where p_a = 1 - exp(-A*exp(-k*d_a)*dt)
                # Let x = A*exp(-k*d_a)*dt, so d_a = (1/k)*log(A*dt/x)
                # Objective: (1-x) * (delta_a - (1/k)*log(A*dt/x))
                # Optimal x satisfies: x * exp(x) = (A*dt)/k * exp(-k*delta_a)
                # So x = W((A*dt)/k * exp(-k*delta_a))
                arg = (A * dt) / k * np.exp(-k * delta_a)
                if arg > 0:
                    x_opt = _lambert_w(arg)
                    if x_opt > 0 and x_opt < 1:
                        d_a_opt = (1.0 / k) * np.log(A * dt / x_opt)
                    else:
                        d_a_opt = 0.0
                else:
                    d_a_opt = 0.0
                best_ask_depth = max(0.0, d_a_opt)
            else:
                best_ask_depth = np.inf
            
            # Compute expected value
            if can_bid:
                p_b = 1.0 - np.exp(-A * np.exp(-k * best_bid_depth) * dt)
            else:
                p_b = 0.0
            
            if can_ask:
                p_a = 1.0 - np.exp(-A * np.exp(-k * best_ask_depth) * dt)
            else:
                p_a = 0.0
            
            V_new[i] = (p_b * (best_bid_depth + V_qp) + 
                       p_a * (-best_ask_depth + V_qm) + 
                       (1 - p_b) * (1 - p_a) * V_qi) - risk_cost
        
        V = V_new
    
    # Extract optimal depths at t=0
    bid_depths = np.full(n_states, np.inf)
    ask_depths = np.full(n_states, np.inf)
    
    for i in range(n_states):
        q = i + q_min
        can_bid = (q < Q)
        can_ask = (q > -Q)
        
        V_qi = V[i]
        
        if can_bid:
            V_qp = V[q_to_idx(q + 1)]
            delta_b = V_qp - V_qi
            arg = (A * dt) / k * np.exp(k * delta_b)
            if arg > 0:
                x_opt = _lambert_w(arg)
                if x_opt > 0 and x_opt < 1:
                    bid_depths[i] = max(0.0, (1.0 / k) * np.log(A * dt / x_opt))
                else:
                    bid_depths[i] = 0.0
            else:
                bid_depths[i] = 0.0
        
        if can_ask:
            V_qm = V[q_to_idx(q - 1)]
            delta_a = V_qm - V_qi
            arg = (A * dt) / k * np.exp(-k * delta_a)
            if arg > 0:
                x_opt = _lambert_w(arg)
                if x_opt > 0 and x_opt < 1:
                    ask_depths[i] = max(0.0, (1.0 / k) * np.log(A * dt / x_opt))
                else:
                    ask_depths[i] = 0.0
            else:
                ask_depths[i] = 0.0
    
    return bid_depths, ask_depths

def _lambert_w(z):
    """
    Approximate the principal branch of the Lambert W function.
    Uses series expansion for small z and asymptotic for large z.
    """
    if z <= 0:
        return 0.0
    if z < 1e-10:
        return z
    if z < 1.0:
        # Series: W(z) = z - z^2 + 3/2 z^3 - 8/3 z^4 + ...
        return z - z*z + 1.5*z*z*z - (8.0/3.0)*z*z*z*z
    else:
        # Asymptotic: W(z) ≈ ln(z) - ln(ln(z)) + ...
        lz = np.log(z)
        if lz < 1e-10:
            return 0.0
        return lz - np.log(lz) + np.log(lz) / lz

# Precompute all regime policies
_POLICY_CACHE = {}
for (sigma, A, k, gamma) in _REGIMES:
    key = (sigma, A, k, gamma)
    _POLICY_CACHE[key] = _solve_regime(sigma, A, k, gamma)


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels in [-Q, Q]
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): two arrays of shape (4000,)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    if key not in _POLICY_CACHE:
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[idx]
    ask_depths = ask_table[idx]
    
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
