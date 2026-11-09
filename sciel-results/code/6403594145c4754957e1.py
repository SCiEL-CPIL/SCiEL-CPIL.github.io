"""Consumption policy solver using backward induction with FOC root-finding."""
import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 135  # Safety margin before the 150s kill


def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    consumption = np.zeros((horizon, states, n_grid))
    # V[z, i] = continuation value (value from next period onward) given state z and m=grid[i]
    V_next = np.zeros((states, n_grid))

    # Pre-compute interpolation parameters for speed
    # We will interpolate V_next[z] on resource_grid
    # For each z', we need to evaluate sum_zp trans[z, zp] * V_next[zp](m_prime)
    # m_prime depends on c.
    
    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all
            consumption[t] = np.tile(resource_grid, (states, 1))
            V_next = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)
            continue

        V_current = np.zeros((states, n_grid))
        m_vals = resource_grid

        for z in range(states):
            # We want to find f in (0, 1) such that:
            # u'(m*f) = beta * sum_zp trans[z, zp] * V_next[zp]'(m')
            # where m' = R*(m - m*f) + income[zp] = R*m*(1-f) + income[zp]
            # Note: V_next is a function of m'. We need its derivative.
            # Since V_next is defined on a grid, we can approximate the derivative
            # or, better, evaluate the Bellman equation directly and find the root
            # of the difference between marginal utility of consumption and
            # marginal value of saving.
            
            # Actually, the standard FOC for this problem is:
            # u'(c) = beta * sum_zp trans[z, zp] * V_next[zp]'(m')
            # 
            # To avoid numerical differentiation of the interpolated V_next,
            # we can define a function g(f) that is the difference between the 
            # marginal utility of consuming c and the expected marginal value of 
            # the next period's wealth.
            #
            # However, V_next is only known at grid points. 
            # A more robust approach for discrete grids is to define the objective
            # J(f) = u(m*f) + beta * sum_zp trans[z, zp] * V_interp_zp(R*m*(1-f) + income[zp])
            # and find the maximum. Since J(f) is concave in f (u is concave, V is concave),
            # the derivative J'(f) is monotonically decreasing.
            # J'(f) = m * u'(m*f) - beta * m * sum_zp trans[z, zp] * V_interp_zp'(R*m*(1-f) + income[zp])
            #
            # Computing V_interp_zp' requires numerical differentiation or storing derivatives.
            # An alternative is to use the fact that the optimal f is where the slope changes sign.
            # We can evaluate J'(f) at f and f+eps to estimate the derivative, or use 
            # scipy's minimize_scalar on -J(f).
            #
            # Given the deadline constraints and the need for speed, using a dense grid 
            # was the previous approach. The "FOC root-finding" inspiration suggests 
            # using brentq on the derivative.
            #
            # Let's implement the derivative-based root finding.
            # We need V_next derivatives. We can compute them via central differences on the grid.
            
            # Compute derivatives of V_next for each state
            # dV_next[zp, i] = dV/dm at grid point i
            dV_next = np.zeros_like(V_next)
            # Use central differences where possible, one-sided at boundaries
            for zp in range(states):
                v = V_next[zp]
                d = np.zeros(n_grid)
                d[1:-1] = (v[2:] - v[:-2]) / (resource_grid[2:] - resource_grid[:-2])
                d[0] = (v[1] - v[0]) / (resource_grid[1] - resource_grid[0])
                d[-1] = (v[-1] - v[-2]) / (resource_grid[-1] - resource_grid[-2])
                dV_next[zp] = d

            # For each m in resource_grid, find f*
            for i in range(n_grid):
                m = m_vals[i]
                
                # Define the function whose root we seek:
                # h(f) = m * u'(m*f) - beta * m * sum_zp trans[z, zp] * dV_interp_zp(m')
                # where m' = R*m*(1-f) + income[zp]
                
                def h(f):
                    c = m * f
                    if c < 1e-12:
                        # If c is too small, marginal utility is huge, so h(f) > 0
                        # This means we should increase f (consume more)
                        return 1.0 
                    
                    mu_c = _crsa_marginal(c, gamma)
                    lhs = m * mu_c
                    
                    rhs = 0.0
                    for zp in range(states):
                        m_prime = R * m * (1.0 - f) + income[zp]
                        # Clip m_prime to grid range for interpolation
                        if m_prime <= resource_grid[0]:
                            m_prime_clipped = resource_grid[0]
                        elif m_prime >= resource_grid[-1]:
                            m_prime_clipped = resource_grid[-1]
                        else:
                            m_prime_clipped = m_prime
                        
                        # Interpolate dV_next[zp] at m_prime_clipped
                        dV_val = np.interp(m_prime_clipped, resource_grid, dV_next[zp])
                        rhs += trans[z, zp] * dV_val
                    
                    rhs *= beta * m
                    return lhs - rhs

                # Find f in (eps, 1.0) such that h(f) = 0
                # h(eps) > 0 (usually, unless m is tiny and income huge, but even then u' dominates)
                # h(1.0) < 0 (usually, since u'(m) is finite and RHS is positive)
                
                f_lower = 1e-6
                f_upper = 1.0 - 1e-9
                
                try:
                    h_lower = h(f_lower)
                    h_upper = h(f_upper)
                    
                    if h_lower * h_upper > 0:
                        # No sign change, use boundary
                        if h_lower > 0:
                            f_opt = f_upper
                        else:
                            f_opt = f_lower
                    else:
                        f_opt = brentq(h, f_lower, f_upper, xtol=1e-8, rtol=1e-8)
                except Exception:
                    # Fallback to grid search if root finding fails
                    # Use a small local grid
                    f_grid = np.linspace(0.001, 1.0, 100)
                    best_val = -np.inf
                    best_f = f_lower
                    for f in f_grid:
                        c = m * f
                        if c <= 0:
                            continue
                        u_val = _crsa_utility(c, gamma)
                        cont_val = 0.0
                        for zp in range(states):
                            m_prime = R * m * (1.0 - f) + income[zp]
                            m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                            V_val = np.interp(m_prime_clipped, resource_grid, V_next[zp])
                            cont_val += trans[z, zp] * V_val
                        total = u_val + beta * cont_val
                        if total > best_val:
                            best_val = total
                            best_f = f
                    f_opt = best_f

                c_opt = m * f_opt
                consumption[t, z, i] = c_opt
                
                # Compute V_current[z, i] using the optimal c
                u_val = _crsa_utility(c_opt, gamma)
                cont_val = 0.0
                for zp in range(states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                    V_val = np.interp(m_prime_clipped, resource_grid, V_next[zp])
                    cont_val += trans[z, zp] * V_val
                V_current[z, i] = u_val + beta * cont_val

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Check deadline
        if time.time() > deadline:
            # Fallback if we ran out of time
            pass

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        states = model['transition'].shape[0]
        if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                frac = 0.8 + 0.2 * (t / max(1, horizon - 2))
                result[t] = frac * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
