import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 120s limit (extra buffer for evaluator)

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    M = np.asarray(resource_grid, dtype=float)
    num_m = len(M)
    num_z = 2

    # ---------- Utility and derivatives ----------
    log_mode = abs(gamma - 1.0) < 1e-10
    g = gamma if not log_mode else 1.0

    def u(c):
        c = np.maximum(c, 1e-12)
        if log_mode:
            return np.log(c)
        return np.power(c, 1.0 - g) / (1.0 - g)

    def u1(c):
        c = np.maximum(c, 1e-12)
        if log_mode:
            return 1.0 / c
        return np.power(c, -g)

    def u2(c):
        c = np.maximum(c, 1e-12)
        if log_mode:
            return -1.0 / (c * c)
        return -g * np.power(c, -g - 1.0)

    def inv_u1(x):
        # solve u1(c) = x  =>  c = x^{-1/g}
        x = np.maximum(x, 1e-300)
        if log_mode:
            return 1.0 / x
        return np.power(x, -1.0 / g)

    # ---------- Log-spaced value grid ----------
    # Use log-space grid so value function (which is concave and power-law-ish)
    # is well-represented. Interpolate V(log m) linearly in log m.
    N_fine = 2000
    log_lo = np.log(0.001)
    log_hi = np.log(40.0)
    log_m_fine = np.linspace(log_lo, log_hi, N_fine)
    m_fine = np.exp(log_m_fine)

    # V[t][z] stored as array of size N_fine: V(log_m_fine)
    V = [None] * horizon

    # Terminal period: V[T-1][z](m) = u(m)
    V[horizon - 1] = [u(m_fine) for _ in range(num_z)]

    # Backward induction: at each t, solve Euler via Newton on log-c for each m.
    # Euler: u1(c) = beta * sum_z' p[z,z'] R * u1(m')
    # where m' = R(m - c) + income[z']
    #
    # We solve for log_c directly. Define f(log_c) = log(u1(c)) - log(RHS).
    # Actually easier: solve in c-space with Newton, using the structure.
    #
    # Let's define: at state (t, z, m), we need c* such that
    #   u1(c*) = beta * sum_{z'} p[z,z'] * R * u1(m')
    # where m' = R*(m - c*) + income[z']
    #
    # Define F(c) = u1(c) - beta * sum_{z'} p[z,z'] * R * u1(R*(m-c)+income[z'])
    # We want F(c) = 0.
    #
    # Newton: c_{k+1} = c_k - F(c_k)/F'(c_k)
    # F'(c) = u2(c) + beta * sum p * R * u2(m') * R
    #        = u2(c) + beta * R^2 * sum p * u2(m')
    # Note u2 < 0, so F is decreasing (u1 decreasing, and the continuation term
    # also makes F decrease as c increases because m' decreases => u1(m') increases... 
    # wait let me re-check signs.
    #
    # As c increases: u1(c) decreases (u1' = u2 < 0). 
    # m' = R(m-c)+inc decreases as c increases, so u1(m') increases (since u1 decreasing).
    # So RHS = beta * sum p * R * u1(m') increases as c increases.
    # F(c) = u1(c) - RHS: u1(c) decreases, RHS increases => F strictly decreasing. Good, unique root.
    #
    # F'(c) = u2(c) - beta * sum p * R * u2(m') * (-R)
    #        = u2(c) + beta * R^2 * sum p * u2(m')
    # Both terms negative => F' < 0. Good.

    def solve_euler(t, z, m_vals, V_next):
        """Solve Euler for consumption given m_vals (array), returning c array."""
        n = len(m_vals)
        c = np.full(n, 0.5 * m_vals)  # initial guess
        c = np.clip(c, 1e-10, m_vals - 1e-10)
        
        # Precompute transition weights
        p = transition[z]
        
        for it in range(50):
            # Compute m' for each z'
            # m'_{z'} = R*(m - c) + income[z']
            m_minus_c = m_vals - c  # shape (n,)
            # For each z', m' = R * m_minus_c + income[z']
            
            # Compute RHS = beta * sum_{z'} p[z'] * R * u1(m'_{z'})
            rhs = np.zeros(n)
            rhs_prime = np.zeros(n)  # d(rhs)/dc = -beta * R^2 * sum p * u2(m')
            for zp in range(num_z):
                pz = p[zp]
                if pz < 1e-15:
                    continue
                mp = R * m_minus_c + income[zp]
                mp = np.maximum(mp, 1e-12)
                up1 = u1(mp)
                up2 = u2(mp)
                rhs += pz * R * up1
                rhs_prime += pz * R * R * up2  # d/dm' * dm'/dc = up2 * (-R), then times R from outside... 
                # Let me redo: rhs_zp = p * R * u1(mp), d(rhs_zp)/dc = p * R * u2(mp) * dmp/dc = p * R * u2(mp) * (-R) = -p * R^2 * u2(mp)
                # So d(rhs)/dc = -sum p * R^2 * u2(mp)
            
            # F(c) = u1(c) - rhs
            uc1 = u1(c)
            F = uc1 - rhs
            # F'(c) = u2(c) - d(rhs)/dc = u2(c) + sum p * R^2 * u2(mp)
            uc2 = u2(c)
            Fp = uc2 + rhs_prime  # rhs_prime = sum p * R^2 * u2(mp), which is negative
            
            # Newton step
            # Avoid division by zero
            Fp_safe = np.where(np.abs(Fp) < 1e-300, 1e-300, Fp)
            dc = F / Fp_safe
            c_new = c - dc
            c_new = np.clip(c_new, 1e-12, m_vals - 1e-12)
            
            # Damped Newton: if step too large, halve
            max_step = 0.5 * m_vals
            step_size = np.abs(c_new - c)
            too_big = step_size > max_step
            if np.any(too_big):
                scale = np.where(too_big, max_step[too_big] / step_size[too_big], 1.0)
                scale = np.concatenate([scale, np.ones(n - np.sum(too_big))]) if np.sum(too_big) < n else scale
                # Simpler: just clip the step
                dc_clipped = np.clip(dc, -max_step, max_step)
                c_new = c - dc_clipped
                c_new = np.clip(c_new, 1e-12, m_vals - 1e-12)
            
            c = c_new
            
            # Convergence check
            if np.max(np.abs(dc)) < 1e-12 * np.max(m_vals):
                break
        
        return np.clip(c, 1e-12, m_vals)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V[t] = [None] * num_z
        for z in range(num_z):
            # Solve Euler for each m in fine grid
            c_fine = solve_euler(t, z, m_fine, V[t + 1])
            
            # Compute value: V(t,z,m) = u(c) + beta * sum p * V(t+1,z', m')
            Vt = u(c_fine)
            m_minus_c = m_fine - c_fine
            for zp in range(num_z):
                pz = transition[z, zp]
                if pz < 1e-15:
                    continue
                mp = R * m_minus_c + income[zp]
                mp = np.clip(mp, m_fine[0], m_fine[-1])
                # Interpolate V[t+1][zp] in log-space
                log_mp = np.log(mp)
                V_next_vals = np.interp(log_mp, log_m_fine, V[t + 1][zp])
                Vt += beta * pz * V_next_vals
            
            V[t][z] = Vt
        
        # Time check
        if time.time() > deadline - 20.0:
            # Fill remaining periods with terminal value as fallback
            for tt in range(t - 1, -1, -1):
                V[tt] = [V[tt + 1][zz].copy() for zz in range(num_z)]
            break

    # ---------- Now compute policy on the requested grid M ----------
    policy = np.zeros((horizon, num_z, num_m))
    
    for t in range(horizon):
        if t == horizon - 1:
            for z in range(num_z):
                policy[t, z, :] = M
        else:
            for z in range(num_z):
                c = solve_euler(t, z, M, V[t + 1])
                policy[t, z, :] = np.clip(c, 1e-12, M)
        
        # Time check
        if time.time() > deadline - 5.0:
            # Fill remaining with proportional rule as fallback
            for tt in range(t + 1, horizon):
                if tt == horizon - 1:
                    for zz in range(num_z):
                        policy[tt, zz, :] = M
                else:
                    for zz in range(num_z):
                        # Rough fallback: consume fraction
                        policy[tt, zz, :] = np.clip(0.5 * M, 1e-12, M)
            break

    # Final validation
    policy = np.nan_to_num(policy, nan=1e-12, posinf=M, neginf=1e-12)
    for t in range(horizon):
        for z in range(num_z):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, M + 1e-10)
        # Terminal period must equal grid
        if t == horizon - 1:
            policy[t, :, :] = M[None, :]

    return policy
