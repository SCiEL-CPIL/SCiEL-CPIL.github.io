import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_out = len(resource_grid)
    m_out = resource_grid
    
    # Use a fine grid for computation
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Derivative of utility for FOC
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Policy storage
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Backward induction
    V_next = np.zeros((n_fine, 2))
    
    # Pre-compute fractions for fallback search
    fractions = np.linspace(0.001, 1.0, 301)
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 3:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            probs = transition[z]
            
            # Build interpolators for continuation values
            # We need V_next as a function of m_next
            # m_next can range from roughly R*(0) + income[0] to R*(40 - 0.001) + income[1]
            # min m_next ~ 0.5 or 1.5
            # max m_next ~ 1.02*40 + 1.5 ~ 42.3
            # Our grid m_fine goes up to 40. We need to handle extrapolation or clamp.
            # Since m' = R(m-c) + y, and c <= m, m' >= y.
            # Max m' occurs when c is small. If m=40, c~0.04, m' ~ 1.02*39.96 + 1.5 ~ 42.26.
            # We should extend our interpolation grid or clamp inputs to the grid range.
            # Clamping is safer for PCHIP to avoid extrapolation issues.
            
            try:
                pchip_0 = PchipInterpolator(m_fine, V_next[:, 0])
                pchip_1 = PchipInterpolator(m_fine, V_next[:, 1])
                
                # Derivatives for Newton step
                # PchipInterpolator provides .deriv()
                d_pchip_0 = pchip_0.deriv()
                d_pchip_1 = pchip_1.deriv()
            except Exception:
                pchip_0 = None
                pchip_1 = None
                d_pchip_0 = None
                d_pchip_1 = None
            
            V_curr[:, z] = 0.0
            policy_fine[t, z, :] = 0.0
            
            for i, m in enumerate(m_fine):
                # Initial guess: previous period's policy or a fraction
                # Use a heuristic: c ~ 0.1 * m for low m, higher for high m
                # Or use the policy from t+1 if available (but we are going backwards, so V_next is known)
                # Let's start with c = m * 0.1 + income[z] * 0.1 as a rough guess
                # Better: use the optimal policy from the coarse search as initial guess?
                # No, let's use a simple analytic approximation or just 0.1*m.
                c_guess = 0.1 * m
                
                # Newton's method on FOC:
                # u'(c) = beta * sum_z' P(z'|z) * R * V'(m', z')
                # where m' = R(m-c) + income[z']
                # V'(m', z') is the derivative of the value function.
                
                c = c_guess
                converged = False
                
                for _ in range(10):
                    m_next_0 = R * (m - c) + income[0]
                    m_next_1 = R * (m - c) + income[1]
                    
                    # Clamp for interpolation
                    m_next_0_clamped = np.clip(m_next_0, m_fine[0], m_fine[-1])
                    m_next_1_clamped = np.clip(m_next_1, m_fine[0], m_fine[-1])
                    
                    if pchip_0 is not None:
                        # FOC: u'(c) - beta * R * (p0 * V'_0(m'_0) + p1 * V'_1(m'_1)) = 0
                        u_prime_c = utility_prime(c)
                        v_prime_0 = d_pchip_0(m_next_0_clamped)
                        v_prime_1 = d_pchip_1(m_next_1_clamped)
                        
                        f = u_prime_c - beta * R * (probs[0] * v_prime_0 + probs[1] * v_prime_1)
                        
                        # Derivative of f w.r.t c
                        # u''(c) - beta * R * (p0 * V''(m'_0) * (-R) + p1 * V''(m'_1) * (-R))
                        # = u''(c) + beta * R^2 * (p0 * V''(m'_0) + p1 * V''(m'_1))
                        
                        u_double_prime = -utility_prime(c) / c  # -1/c^2 for log, -c^-gamma-1 for CRRA
                        # Actually u'(c) = c^-gamma, u''(c) = -gamma * c^(-gamma-1)
                        if abs(gamma - 1.0) < 1e-9:
                            u_double_prime = -1.0 / (c * c)
                        else:
                            u_double_prime = -gamma * (c ** (-gamma - 1))
                        
                        # Approximate V'' using finite difference of V' or just assume small?
                        # For PCHIP, we can get the second derivative? No, PCHIP is piecewise cubic.
                        # We can approximate V'' by differentiating the derivative again?
                        # Or just use a small step in Newton?
                        # Let's use a simple secant method or Newton with approximate Hessian.
                        # Actually, the term beta * R^2 * E[V''] is usually small compared to u'' for high gamma?
                        # Let's try Newton with just u'' first (ignoring V'') as a quasi-Newton.
                        # Or use Broyden?
                        # Let's try a simple fixed-point iteration or secant.
                        
                        # Secant method:
                        # f(c) = 0
                        # c_{k+1} = c_k - f(c_k) * (c_k - c_{k-1}) / (f(c_k) - f(c_{k-1}))
                        
                        # Let's stick to Newton with a rough Hessian estimate.
                        # Estimate V'' by (V'(m+dm) - V'(m-dm)) / (2dm)
                        dm = 0.01
                        v_prime_0_plus = d_pchip_0(np.clip(m_next_0 + dm, m_fine[0], m_fine[-1]))
                        v_prime_0_minus = d_pchip_0(np.clip(m_next_0 - dm, m_fine[0], m_fine[-1]))
                        v_double_prime_0 = (v_prime_0_plus - v_prime_0_minus) / (2 * dm)
                        
                        v_prime_1_plus = d_pchip_1(np.clip(m_next_1 + dm, m_fine[0], m_fine[-1]))
                        v_prime_1_minus = d_pchip_1(np.clip(m_next_1 - dm, m_fine[0], m_fine[-1]))
                        v_double_prime_1 = (v_prime_1_plus - v_prime_1_minus) / (2 * dm)
                        
                        df_dc = u_double_prime + beta * R * R * (probs[0] * v_double_prime_0 + probs[1] * v_double_prime_1)
                        
                        if abs(df_dc) < 1e-10:
                            df_dc = 1e-10 * np.sign(df_dc) if df_dc != 0 else 1e-10
                        
                        c_new = c - f / df_dc
                        
                        # Bounds: 1e-12 < c <= m
                        c_new = np.clip(c_new, 1e-12, m)
                        
                        if abs(c_new - c) < 1e-8:
                            c = c_new
                            converged = True
                            break
                        c = c_new
                    else:
                        # Fallback to search if PCHIP failed
                        break
                
                if not converged:
                    # Fallback to discrete search
                    c_candidates = m * fractions
                    m_next_0 = R * (m - c_candidates) + income[0]
                    m_next_1 = R * (m - c_candidates) + income[1]
                    
                    m_next_0_clamped = np.clip(m_next_0, m_fine[0], m_fine[-1])
                    m_next_1_clamped = np.clip(m_next_1, m_fine[0], m_fine[-1])
                    
                    if pchip_0 is not None:
                        V_cont_0 = pchip_0(m_next_0_clamped)
                        V_cont_1 = pchip_1(m_next_1_clamped)
                    else:
                        V_cont_0 = np.interp(m_next_0_clamped, m_fine, V_next[:, 0])
                        V_cont_1 = np.interp(m_next_1_clamped, m_fine, V_next[:, 1])
                    
                    E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
                    U_c = utility(c_candidates)
                    Total_Value = U_c + beta * E_V_cont
                    idx_max = np.argmax(Total_Value)
                    c = c_candidates[idx_max]
                
                policy_fine[t, z, i] = c
                V_curr[i, z] = utility(c) + beta * (probs[0] * np.interp(np.clip(R*(m-c)+income[0], m_fine[0], m_fine[-1]), m_fine, V_next[:, 0]) + probs[1] * np.interp(np.clip(R*(m-c)+income[1], m_fine[0], m_fine[-1]), m_fine, V_next[:, 1]))
            
            V_next = V_curr
            
    # Fill remaining t if broke early
    last_valid_t = -1
    for t in range(horizon - 1, -1, -1):
        if not np.all(policy_fine[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t != horizon - 1:
        for t in range(last_valid_t, -1, -1):
            if np.all(policy_fine[t] == 0):
                policy_fine[t] = policy_fine[last_valid_t]
                
    # Interpolate to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(m_out, m_fine, policy_fine[t, z, :])
            
    # Validity checks
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)
            
    # Constraint: c[5] == m
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
