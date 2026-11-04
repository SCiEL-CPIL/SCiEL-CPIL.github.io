        res = minimize(
            _soft_value, x0, args=(n, 1000.0),
            jac=lambda x, n, mu: _soft_grad(x, n, mu),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
