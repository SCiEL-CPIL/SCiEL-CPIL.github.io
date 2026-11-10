import time
import numpy as np

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])

    grid = np.asarray(resource_grid, dtype=float)
    N = len(grid)
    Z = len(income)
    T = horizon

    # Policy array c[t, z, i]
    c_policy = np.zeros((T, Z, N))

    # Last period: consume everything (within 1e-9 of grid)
    c_policy[T - 1] = grid[np.newaxis, :]

    def is_log():
        return abs(gamma - 1.0) < 1e-9

    def util(c):
        c = np.maximum(c, 1e-12)
        if is_log():
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def u1(c):
        c = np.maximum(c, 1e-12)
        if is_log():
            return 1.0 / c
        return np.power(c, -gamma)

    def u2(c):
        c = np.maximum(c, 1e-12)
        if is_log():
            return -1.0 / (c * c)
        return -gamma * np.power(c, -gamma - 1.0)

    # Continuation values V[t, z, i] for t in 0..T (V[T]=0)
    V = np.zeros((T + 1, Z, N))
    # V[T] = 0

    # Work backward
    for t in range(T - 2, -1, -1):
        if time.time() > DEADLINE:
            # Return what we have so far (fill remaining with a heuristic)
            break
        # For each income state z
        for z in range(Z):
            m = grid  # (N,)
            # For each next state z'
            # m' = R*(m - c) + income[z']
            # continuation value at (t+1, z') as function of m'
            # V_next[z'] = V[t+1, z']  (N,)
            # We need to interpolate V[t+1, z'] at m' which may be outside grid.
            # Build interpolant of V_next over grid, with linear extrapolation.

            # We'll solve for c via Euler:
            # u1(c) = beta * sum_{z'} p[z,z'] * R * u1(m')
            # where m' = R*(m-c)+income[z']

            # Newton on c for each grid point.
            c = np.empty(N)
            # Initial guess: proportional saving rule.
            # Use a simple initial guess: c = m * 0.5 clipped.
            c = np.clip(0.5 * m, 1e-9, m)

            # Precompute interpolants for each z' of V[t+1, z']
            # We'll use np.interp per z' on the fly (vectorized over m').
            Vnext = V[t + 1]  # (Z, N)

            # Newton iterations
            for it in range(60):
                # Compute m' for each z'
                # m'_{z'} = R*(m - c) + income[z']   shape (Z, N)
                mp = R * (m[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
                # Clip mp to >= 0 for interpolation sanity (but can be negative if c>m)
                # Interpolate V[t+1, z'] at mp
                # np.interp requires increasing xp; grid is increasing.
                # For each z', interpolate Vnext[z'] at mp[z']
                Vmp = np.empty((Z, N))
                for zp in range(Z):
                    Vmp[zp] = np.interp(mp[zp], grid, Vnext[zp],
                                        left=Vnext[zp, 0], right=Vnext[zp, -1])
                p = transition[z]  # (Z,)
                # expected continuation
                Vexp = np.sum(p[:, np.newaxis] * Vmp, axis=0)  # (N,)

                # RHS of Euler
                rhs = beta * R * np.sum(p[:, np.newaxis] * u1(mp), axis=0)  # (N,)
                lhs = u1(c)

                f = lhs - rhs
                # derivative of f w.r.t c:
                # d/dc lhs = u2(c)
                # d/dc rhs = beta * R * sum p * d/dc u1(mp)
                # d/dc u1(mp) = u2(mp) * dmp/dc = u2(mp) * (-R)
                dmp_dc = -R
                drhs_dc = beta * R * np.sum(p[:, np.newaxis] * (u2(mp) * dmp_dc), axis=0)
                df_dc = u2(c) - drhs_dc

                # Newton step
                step = f / df_dc
                c_new = c - step
                # Enforce 0 < c <= m + 1e-10
                c_new = np.clip(c_new, 1e-9, m + 1e-10)
                # Damped if it overshoots
                if np.all(np.isfinite(c_new)):
                    c = c_new
                if np.max(np.abs(step)) < 1e-12:
                    break

            c_policy[t, z] = c

            # Compute V[t, z] = util(c) + beta * E[V[t+1, z'](m')]
            # Recompute Vexp with final c
            mp = R * (m[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
            Vmp = np.empty((Z, N))
            for zp in range(Z):
                Vmp[zp] = np.interp(mp[zp], grid, Vnext[zp],
                                    left=Vnext[zp, 0], right=Vnext[zp, -1])
            p = transition[z]
            Vexp = np.sum(p[:, np.newaxis] * Vmp, axis=0)
            V[t, z] = util(c) + beta * Vexp

    # Ensure final period still matches grid
    c_policy[T - 1] = grid[np.newaxis, :]

    # Final validity pass: clip to (0, m+1e-10]
    m = grid[np.newaxis, np.newaxis, :]
    c_policy = np.clip(c_policy, 1e-9, m + 1e-10)
    c_policy = np.nan_to_num(c_policy, nan=1e-9, posinf=1.0, neginf=1e-9)

    return c_policy
# EVOLVE-BLOCK-END
