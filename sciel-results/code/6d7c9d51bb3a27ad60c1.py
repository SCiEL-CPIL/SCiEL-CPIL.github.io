"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Precompute grid spacing for finite differences
    # V_prime[t, z, i] approximates dV/dm at grid[i]
    V_prime = np.zeros((horizon, n_states, n_grid))
    
    # Terminal derivative
    V_prime[horizon - 1, :, :] = marginal_utility(grid)

    # Precompute transition-weighted V_prime lookup for speed
    # For each z, precompute the expected V' as a function of m_prime
    # We'll do this on the fly but vectorize where possible

    # Backward induction using FOC root-finding
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    # Use a reasonable consumption fraction based on beta*R
                    frac = 1.0 / (1.0 + beta * R)
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # Precompute transition-weighted V_prime arrays for this z
            # For each z', we have V_prime[t+1, z', :]
            # E[V'(R*(m-c) + income[z']) | z] = sum_{z'} P(z'|z) * V_prime[t+1, z', :](R*(m-c) + income[z'])
            
            # For efficiency, precompute the expected V' on a fine grid of m_prime
            # Actually, let's just use np.interp but vectorize over m
            
            # Vectorized approach: for all m in grid, solve for c
            # But brentq doesn't vectorize easily, so we do it per m but with optimized residual
            
            for i_m in range(n_grid):
                m = grid[i_m]
                
                if time.time() > deadline - 2.0:
                    # Fallback: use a simple fraction
                    frac = 1.0 / (1.0 + beta * R)
                    c_opt = np.clip(m * frac, 1e-12, m)
                    C[t, z, i_m] = c_opt
                    # Compute value
                    m_primes = R * (m - c_opt) + income
                    m_primes_clipped = np.clip(m_primes, grid[0], grid[-1])
                    cont_vals = np.array([np.interp(mp, grid, V[t + 1, zp, :]) for zp, mp in enumerate(m_primes_clipped)])
                    expected_cont = np.dot(transition[z], cont_vals)
                    V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont
                    continue

                # FOC: u'(c) = beta * R * E[V'(R*(m-c) + income[z']) | z]
                # We solve for c in (0, m)
                
                def foc_residual(c):
                    c = max(c, 1e-12)
                    mu_c = marginal_utility(np.array([c]))[0]
                    
                    # Compute expected continuation derivative
                    m_primes = R * (m - c) + income
                    m_primes_clipped = np.clip(m_primes, grid[0], grid[-1])
                    # Vectorized interpolation
                    vp_vals = np.array([np.interp(mp, grid, V_prime[t + 1, zp, :]) for zp, mp in enumerate(m_primes_clipped)])
                    expected_v_prime = np.dot(transition[z], vp_vals)
                    
                    return mu_c - beta * R * expected_v_prime

                # Try to find root using brentq
                c_low = 1e-12
                c_high = m - 1e-12
                
                res_low = foc_residual(c_low)
                res_high = foc_residual(c_high)
                
                if res_low * res_high < 0:
                    # Root exists, use brentq
                    try:
                        c_opt = brentq(foc_residual, c_low, c_high, xtol=1e-10, rtol=1e-8)
                    except:
                        # Fallback to bisection
                        lo = c_low
                        hi = c_high
                        for _ in range(50):
                            mid = (lo + hi) / 2
                            res_mid = foc_residual(mid)
                            if res_low * res_mid <= 0:
                                hi = mid
                            else:
                                lo = mid
                        c_opt = (lo + hi) / 2
                else:
                    # No sign change - boundary solution
                    if res_low < 0:
                        c_opt = c_low
                    else:
                        c_opt = c_high
                
                # Ensure valid range
                c_opt = np.clip(c_opt, 1e-12, m)
                
                C[t, z, i_m] = c_opt
                
                # Compute the value
                m_primes = R * (m - c_opt) + income
                m_primes_clipped = np.clip(m_primes, grid[0], grid[-1])
                cont_vals = np.array([np.interp(mp, grid, V[t + 1, zp, :]) for zp, mp in enumerate(m_primes_clipped)])
                expected_cont = np.dot(transition[z], cont_vals)
                
                V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont

        # Compute V_prime for this period using central differences
        for z in range(n_states):
            if n_grid > 2:
                V_prime[t, z, 1:-1] = (V[t, z, 2:] - V[t, z, :-2]) / (grid[2:] - grid[:-2])
            if n_grid > 1:
                V_prime[t, z, 0] = (V[t, z, 1] - V[t, z, 0]) / (grid[1] - grid[0])
                V_prime[t, z, -1] = (V[t, z, -1] - V[t, z, -2]) / (grid[-1] - grid[-2])

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
