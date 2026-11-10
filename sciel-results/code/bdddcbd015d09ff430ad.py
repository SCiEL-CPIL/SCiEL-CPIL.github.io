import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    num_states = len(income)
    c_floor = 1e-12
    
    # Use a denser internal grid for backward induction
    n_internal = 400
    m_internal = np.geomspace(0.001, 40, n_internal)
    log_m_internal = np.log(m_internal)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Utility second derivative
    def u_double_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1)
    
    # Backward induction on internal grid
    policy_int = np.zeros((horizon, num_states, n_internal))
    value_int = np.zeros((horizon, num_states, n_internal))
    
    # Terminal period
    t = horizon - 1
    for z in range(num_states):
        value_int[t, z, :] = utility(m_internal)
        policy_int[t, z, :] = m_internal
    
    # Build interpolation functions for V' at t+1
    def build_vp_interp(zp, t_idx):
        """Build PCHIP interpolator for V' at time t_idx, state zp"""
        vp_vals = u_prime(policy_int[t_idx, zp, :])
        try:
            return PchipInterpolator(m_internal, vp_vals)
        except Exception:
            return None
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: consume all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy_int[tt, z, :] = m_internal
                    value_int[tt, z, :] = utility(m_internal)
            break
        
        # Pre-build V' interpolators for t+1
        vp_interps = []
        val_interps = []
        for zp in range(num_states):
            vp_interps.append(build_vp_interp(zp, t + 1))
            try:
                val_interps.append(PchipInterpolator(m_internal, value_int[t + 1, zp, :]))
            except Exception:
                val_interps.append(None)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Newton-Raphson on FOC: f(c) = u'(c) - beta*R*sum P(z')*V'(m') = 0
            # f is decreasing in c (u' decreasing, V' increasing as m' decreases)
            c = np.maximum(0.5 * m_internal, c_floor)  # initial guess: half of wealth
            
            max_iter = 30
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Evaluate FOC at current c
                lhs = u_prime(c)
                rhs = np.zeros(n_internal)
                rhs_prime = np.zeros(n_internal)  # derivative of rhs w.r.t. c
                
                for zp in range(num_states):
                    if probs[zp] < 1e-15:
                        continue
                    m_next = R * (m_internal - c) + income[zp]
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                    
                    if vp_interps[zp] is not None:
                        vp_vals = vp_interps[zp](m_next_clipped)
                    else:
                        vp_vals = np.interp(m_next_clipped, m_internal, u_prime(policy_int[t+1, zp, :]))
                    rhs += probs[zp] * vp_vals
                    # For Newton, we need d/dc of rhs. Since m_next depends on c,
                    # d/dc [V'(m_next)] = V''(m_next) * d/dc[m_next] = V''(m_next) * (-R)
                    # But we don't have V'' easily. Use finite difference or just bisection fallback.
                    # Let's use bisection for robustness, but with better initial guess.
                
                rhs = beta * R * rhs
                
                # f(c) = lhs - rhs; f decreasing
                f_val = lhs - rhs
                too_high = f_val > 0  # c too low, need to increase c
                
                # Use bisection with better bracketing
                c_low = np.where(too_high, c, np.maximum(0.5 * c, c_floor))
                c_high = np.where(~too_high, c, np.minimum(1.5 * c, m_internal))
                c_low = np.clip(c_low, c_floor, m_internal)
                c_high = np.clip(c_high, c_floor, m_internal)
                
                c_new = (c_low + c_high) / 2.0
                c = c_new
                c = np.clip(c, c_floor, m_internal)
                
                if np.all(c_high - c_low < 1e-12 * m_internal):
                    break
            
            policy_int[t, z, :] = np.clip(c, c_floor, m_internal)
            
            # Compute value
            u_val = utility(policy_int[t, z, :])
            cont_val = np.zeros(n_internal)
            for zp in range(num_states):
                if probs[zp] < 1e-15:
                    continue
                m_next = R * (m_internal - policy_int[t, z, :]) + income[zp]
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                
                if val_interps[zp] is not None:
                    v_next = val_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, m_internal, value_int[t + 1, zp, :])
                cont_val += probs[zp] * v_next
            
            value_int[t, z, :] = u_val + beta * cont_val
    
    # Interpolate policy back to the original resource_grid
    policy = np.zeros((horizon, num_states, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_states):
            # Use PCHIP for smooth interpolation
            try:
                interp = PchipInterpolator(m_internal, policy_int[t, z, :])
                policy[t, z, :] = interp(resource_grid)
            except Exception:
                policy[t, z, :] = np.interp(resource_grid, m_internal, policy_int[t, z, :])
            # Enforce constraints
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, resource_grid)
    
    # Ensure terminal condition is exact
    for z in range(num_states):
        policy[horizon - 1, z, :] = resource_grid
    
    return policy
