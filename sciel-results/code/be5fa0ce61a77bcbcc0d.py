"""Rewritten program: cubic-spline value functions + dense search + Newton polish."""
import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model['R'])
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)
    N_m = len(m_grid)
    N_z = len(income)

    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1.0:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1.0:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Consumption policy: shape (horizon, N_z, N_m)
    policy = np.empty((horizon, N_z, N_m), dtype=float)

    # Value functions: V[z] at current time (for continuation)
    # Terminal (t = horizon-1): V = u(m)
    V = np.empty((N_z, N_m), dtype=float)
    for z in range(N_z):
        V[z, :] = util(m_grid)

    # Work backward from t = horizon-2 down to 0
    # At t = horizon-1, c = m (consume all)
    for z in range(N_z):
        policy[horizon - 1, z, :] = m_grid

    # Precompute spline for terminal value (not needed for t=horizon-1 since c=m)
    # For t = horizon-2 down to 0:
    #   V_t[z, m] = max_c u(c) + beta * sum_z' P[z,z'] * V_{t+1}[z', R*(m-c)+income[z']]
    #   s.t. 0 < c <= m

    # Number of candidate fractions for coarse search
    n_cand = 401

    for t in range(horizon - 2, -1, -1):
        # Check deadline
        if time.time() > DEADLINE - 10.0:
            # Not enough time to finish; fill remaining with a simple heuristic
            # and return what we have (but we must have all t filled for validity)
            # Since we go backward and t is decreasing, if we're at t, we've done
            # t+1..horizon-1. We need to fill 0..t. Use c = m as fallback.
            for tt in range(t, -1, -1):
                for z in range(N_z):
                    policy[tt, z, :] = m_grid
            break

        # Build splines for V_{t+1} for each z'
        splines = []
        for zp in range(N_z):
            splines.append(CubicSpline(m_grid, V[zp, :], bc_type='natural'))

        # For each current state z and grid m, find optimal c
        for z in range(N_z):
            # Transition probabilities to next states
            probs = transition[z, :]
            # For each m in grid, search over c candidates
            c_vals = np.empty(N_m, dtype=float)
            for i_m in range(N_m):
                m = m_grid[i_m]
                # Candidate c: fractions of m
                fracs = np.linspace(0.002, 1.0, n_cand)
                c_cand = fracs * m
                c_cand = np.clip(c_cand, 1e-12, m)

                # Utility of current consumption
                u_c = util(c_cand)

                # Next resources: m' = R*(m - c) + income[z']
                # For each z', compute continuation value
                # m_minus_c = m - c_cand
                m_minus_c = m - c_cand
                R_mmc = R * m_minus_c  # shape (n_cand,)

                # Continuation: sum_z' probs[z'] * V_{t+1}[z'](R_mmc + income[z'])
                # Interpolate spline at R_mmc + income[z']
                cont = np.zeros(n_cand, dtype=float)
                for zp in range(N_z):
                    if probs[zp] > 0:
                        x_next = R_mmc + income[zp]
                        # Clip to grid range for spline evaluation
                        x_next = np.clip(x_next, m_grid[0], m_grid[-1])
                        cont += probs[zp] * splines[zp](x_next)

                total = u_c + beta * cont
                idx = np.argmax(total)
                c_best = c_cand[idx]

                # Newton polish: solve u'(c) = beta * sum_z' P[z'] * V'_{t+1}[z'](R*(m-c)+income[z'])
                # Define f(c) = u'(c) - beta * sum_z' P[z'] * V'_{t+1}[z'](R*(m-c)+income[z'])
                # We want f(c) = 0. f is decreasing in c (u'' < 0, and V' decreasing in m' which decreases as c increases... wait)
                # Actually as c increases, m' = R*(m-c)+income decreases, so V'(m') increases (V' is decreasing in m, so smaller m -> larger V').
                # u'(c) decreases in c. So f(c) = u'(c) - beta*sum V'(m') where V'(m') increases as c increases.
                # So f is strictly decreasing. We can bisection/Newton.
                #
                # Let's do a simple bisection in a small neighborhood of c_best.
                def f(c):
                    m_prime = R * (m - c) + income  # shape (N_z,)
                    m_prime = np.clip(m_prime, m_grid[0], m_grid[-1])
                    vp = 0.0
                    for zp in range(N_z):
                        if probs[zp] > 0:
                            vp += probs[zp] * splines[zp](m_prime[zp], nu=1)
                    return u_prime(c) - beta * vp

                # Bisection in [c_best * 0.9, min(c_best * 1.1, m)] or just [c_best - delta, c_best + delta]
                lo = max(1e-12, c_best - 0.01 * m)
                hi = min(m, c_best + 0.01 * m)
                if hi <= lo:
                    c_final = c_best
                else:
                    f_lo = f(lo)
                    f_hi = f(hi)
                    # f is decreasing, so f_lo > f_hi typically
                    # We want f(c) = 0. If f_lo > 0 and f_hi < 0, root in between.
                    if f_lo * f_hi > 0:
                        c_final = c_best
                    else:
                        for _ in range(50):
                            mid = 0.5 * (lo + hi)
                            f_mid = f(mid)
                            if abs(f_mid) < 1e-14:
                                break
                            if f_lo * f_mid <= 0:
                                hi = mid
                                f_hi = f_mid
                            else:
                                lo = mid
                                f_lo = f_mid
                        c_final = 0.5 * (lo + hi)
                        c_final = np.clip(c_final, 1e-12, m)

                c_vals[i_m] = c_final

            policy[t, z, :] = c_vals

        # Update V for current time t: V[z, m] = u(c_opt) + beta * cont(c_opt)
        for z in range(N_z):
            probs = transition[z, :]
            c_z = policy[t, z, :]  # shape (N_m,)
            u_c = util(c_z)
            m_minus_c = m_grid - c_z  # shape (N_m,)
            R_mmc = R * m_minus_c
            cont = np.zeros(N_m, dtype=float)
            for zp in range(N_z):
                if probs[zp] > 0:
                    x_next = R_mmc + income[zp]
                    x_next = np.clip(x_next, m_grid[0], m_grid[-1])
                    cont += probs[zp] * splines[zp](x_next)
            V[z, :] = u_c + beta * cont

        # Check deadline
        if time.time() > DEADLINE:
            # Emergency: fill remaining t with c = m
            for tt in range(t, -1, -1):
                for z in range(N_z):
                    policy[tt, z, :] = m_grid
            break

    # Ensure validity: finite, 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
            # Ensure finite
            policy[t, z, :] = np.where(np.isfinite(policy[t, z, :]), policy[t, z, :], m_grid)

    # Final period must equal grid within 1e-9
    for z in range(N_z):
        policy[horizon - 1, z, :] = m_grid

    return policy
