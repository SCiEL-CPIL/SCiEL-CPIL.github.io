    def optimize_full(x0):
        x = np.clip(x0, -0.5, 0.5).copy()
        for alpha, mi, ftol in [(40, 300, 1e-9), (200, 300, 1e-10), (800, 400, 1e-12)]:
            res = minimize(softmin_obj, x, args=(alpha,), jac=True,
                           method='L-BFGS-B',
                           bounds=[(-0.5,0.5)]*(N*3),
                           options={'maxiter': mi, 'maxls': 50, 'ftol': ftol, 'gtol': 1e-8})
            x = res.x
        return x
