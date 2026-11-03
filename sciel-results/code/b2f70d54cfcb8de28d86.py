    def _soft_min_optimize(pos, beta=200.0, max_iter=500, ftol=1e-12, gtol=1e-10):
        """Optimize using soft-min objective."""
        bounds = [(-HALF, HALF)] * (N * 3)
        try:
            res = minimize(
                lambda x: _soft_min_objective_and_grad(x, beta)[0],
                pos.flatten(),
                jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            if res.success or np.isfinite(res.fun):
                new_pos = res.x.reshape(N, 3)
                if np.all(np.isfinite(new_pos)):
                    return new_pos
        except Exception:
            pass
        return pos
    