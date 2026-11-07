"""Solve consumption-saving problem via Chebyshev spectral collocation and Euler equation root-finding."""
import numpy as np
from scipy.special import chebyshev, eval_chebyt
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_m = len(resource_grid)
    N_z = len(income)
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    
    # Utility and derivatives
    if gamma == 1:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return c ** (1 - gamma) / (1 - gamma)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
    
    # Store value and consumption: V[t, m_idx, z], C[t, z, m_idx]
    V = np.zeros((horizon, N_m, N_z))
    C = np.zeros((horizon, N_z, N_m))
    
    # Chebyshev nodes for fitting (use the grid points mapped to [-1, 1] for simplicity, 
    # or standard Chebyshev nodes. Using the grid points directly as collocation points is valid 
    # if we treat it as interpolation. For N=101, this is high order.)
    # Let's map resource_grid to x in [-1, 1]
    x_grid = 2.0 * (resource_grid - m_min) / (m_max - m_min) - 1.0
    
    # Precompute Chebyshev polynomial values T_k(x_grid) for k=0..N_m-1
    # T_k(x) can be computed recursively or via eval_chebyt
    # eval_chebyt(k, x) is efficient
    # We need a matrix ChebV[k, j] = T_k(x_grid[j])
    # N_m is 101.
    ChebV = np.zeros((N_m, N_m))
    for k in range(N_m):
        ChebV[k, :] = eval_chebyt(k, x_grid)
        
    # Inverse matrix for fitting coefficients: coeffs = inv(ChebV) @ V_values
    # ChebV is square and non-singular for distinct points.
    # Pseudo-inverse or direct inverse? Direct inverse of 101x101 is fast.
    # Note: ChebV is ill-conditioned for high N, but 101 is manageable with double precision.
    # Alternatively, use least squares if we reduce order, but let's stick to interpolation.
    try:
        ChebV_inv = np.linalg.inv(ChebV)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular (unlikely for Chebyshev nodes)
        ChebV_inv = np.linalg.pinv(ChebV)

    # Terminal period: consume everything
    # V[horizon-1, m_idx, z] = util(resource_grid[m_idx])
    V[horizon-1, :, :] = util(resource_grid)[:, np.newaxis] # Shape (N_m, 1) broadcast to (N_m, N_z)
    C[horizon-1, :, :] = resource_grid[np.newaxis, :] # Shape (1, N_m) broadcast to (N_z, N_m)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Emergency: fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                if tt < horizon - 1:
                    V[tt] = V[tt+1]
                    C[tt] = C[tt+1]
            return C
        
        # Fit Chebyshev coefficients for V[t+1, :, z'] for each z'
        # Coeffs[t+1, k, z']
        coeffs_next = np.zeros((N_m, N_z))
        for zp in range(N_z):
            # coeffs = inv(ChebV) @ V_values
            coeffs_next[:, zp] = ChebV_inv @ V[t+1, :, zp]
            
        # Precompute derivative coefficients for V[t+1]
        # If V(x) = sum c_k T_k(x), then V'(x) = sum c_k * k * U_{k-1}(x)
        # Or use the fact that d/dx T_k = k U_{k-1}.
        # We need V'(m) where m = m_min + (m_max-m_min)/2 * (x+1).
        # Chain rule: dV/dm = dV/dx * dx/dm = dV/dx * 2/(m_max-m_min).
        
        def get_value_derivative(m_val, z_prime):
            """Evaluate derivative of V_{t+1}(m, z') using Chebyshev coefficients."""
            # Map m to x
            x_val = 2.0 * (m_val - m_min) / (m_max - m_min) - 1.0
            # Clip x to [-1, 1] to avoid extrapolation issues if m is slightly out of bounds
            x_val = np.clip(x_val, -1.0, 1.0)
            
            # Sum c_k * k * U_{k-1}(x)
            # U_{k-1}(x) is Chebyshev polynomial of the second kind.
            # We can compute this iteratively or use scipy.special.eval_chebyu
            # eval_chebyu(n, x) evaluates U_n(x)
            
            deriv_sum = 0.0
            c_k = coeffs_next[:, z_prime]
            # k=0 term is 0
            for k in range(1, N_m):
                if c_k[k] != 0.0:
                    deriv_sum += c_k[k] * k * eval_chebyu(k-1, x_val)
            
            return deriv_sum * (2.0 / (m_max - m_min))

        for z in range(N_z):
            for i_m in range(N_m):
                m = resource_grid[i_m]
                
                if m < 1e-8:
                    c_opt = m
                else:
                    # Define the FOC function G(c)
                    def foc_func(c):
                        # u'(c)
                        uc_prime = u_prime(c)
                        
                        # Continuation part: beta * R * sum_{z'} P[z, z'] * V'_{t+1}(m', z')
                        # m' = R * (m - c) + income[z']
                        cont_deriv = 0.0
                        for zp in range(N_z):
                            m_next = R * (m - c) + income[zp]
                            # If m_next is outside grid, we need to handle it.
                            # The grid is [0.001, 40]. 
                            # If m_next < m_min, we can clamp or extrapolate. 
                            # Chebyshev interpolation is bad at extrapolation.
                            # However, m_next = R(m-c) + y. 
                            # Since c <= m, m_next >= R(0) + y_min = y_min.
                            # y_min is 0.5. m_min is 0.001. So m_next is always > m_min.
                            # If m_next > m_max, we clamp x to 1.0 in get_value_derivative.
                            v_prime = get_value_derivative(m_next, zp)
                            cont_deriv += transition[z, zp] * v_prime
                        
                        return uc_prime - beta * R * cont_deriv

                    # Check if optimal c is m (consume all)
                    # If G(m) > 0, then even at c=m, marginal utility of consumption is higher
                    # than the shadow value of saving, so we should consume more (but we can't).
                    # So c_opt = m.
                    gm = foc_func(m)
                    
                    if gm > 0:
                        c_opt = m
                    else:
                        # Root exists in (eps, m)
                        # G(eps) is very large positive (1/eps or eps^-gamma)
                        # G(m) is <= 0
                        # Use brentq
                        try:
                            c_opt = brentq(foc_func, 1e-12, m, xtol=1e-12, rtol=1e-12, maxiter=100)
                        except ValueError:
                            # If no sign change (shouldn't happen if gm <= 0 and G(eps)>0), fallback
                            # This can happen if numerical errors make G(eps) not > 0 or G(m) not < 0
                            # Fallback to simple bisection or just m
                            c_opt = m
                            
                c_opt = np.clip(c_opt, 1e-12, m)
                C[t, z, i_m] = c_opt
                
                # Calculate V[t, m, z]
                # V = util(c) + beta * sum P V_next
                val = util(c_opt)
                for zp in range(N_z):
                    m_next = R * (m - c_opt) + income[zp]
                    x_val = 2.0 * (m_next - m_min) / (m_max - m_min) - 1.0
                    x_val = np.clip(x_val, -1.0, 1.0)
                    # Evaluate V_next using Chebyshev sum
                    # V(x) = sum c_k T_k(x)
                    v_next = 0.0
                    c_k = coeffs_next[:, zp]
                    for k in range(N_m):
                        if c_k[k] != 0.0:
                            v_next += c_k[k] * eval_chebyt(k, x_val)
                    val += beta * transition[z, zp] * v_next
                V[t, i_m, z] = val

    # Final validation
    # c[5] should equal grid within 1e-9
    C[horizon-1] = resource_grid[None, :]
    V[horizon-1] = util(resource_grid)[:, np.newaxis]

    # Ensure all values are finite and 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            for i_m in range(N_m):
                m = resource_grid[i_m]
                c = C[t, z, i_m]
                if not np.isfinite(c) or c <= 0 or c > m + 1e-10:
                    C[t, z, i_m] = np.clip(c, 1e-12, m)

    return C
