def _refine_slsqp(x0, bounds):
    def objective(x):
        return -np.sum(x[2*N:])

    def boundary_cons(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        # 4*N constraints
        c = np.zeros(4*N)
        c[0:N] = pos[:, 0] - radii
        c[N:2*N] = 1.0 - pos[:, 0] - radii
        c[2*N:3*N] = pos[:, 1] - radii
        c[3*N:4*N] = 1.0 - pos[:, 1] - radii
        return c

    def overlap_cons(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        # 325 constraints
        i, j = np.triu_indices(N, k=1)
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist_sq = dx*dx + dy*dy
        sum_r = radii[i] + radii[j]
        return dist_sq - sum_r*sum_r

    constraints = [
        {'type': 'ineq', 'fun': boundary_cons},
        {'type': 'ineq', 'fun': overlap_cons}
    ]

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-10})
        if res.success and np.isfinite(res.fun):
            return res.x
    except Exception:
        pass
    return x0
