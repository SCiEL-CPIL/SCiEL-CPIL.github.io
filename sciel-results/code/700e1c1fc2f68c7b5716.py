<<<<<<< SEARCH
    # If rank-21 not found, try a few random restarts for rank-21
    if _remaining() > 40.0:
        def _random_rank21_attempt():
            r = 21
            x0 = np.random.randn(nm * r + mp * r + pn * r) * 0.1

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 300, 'gtol': 1e-7})
                if res.fun < 1e-10:
                    x = res.x
                    U = x[:nm * r].reshape(nm, r)
                    V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W = x[nm * r + mp * r:].reshape(pn, r)
                    if _verify_full(U, V, W, n, m, p, tol=1e-6):
                        return U.tolist(), V.tolist(), W.tolist(), r
            except Exception:
                pass
            return None

        for _ in range(3):
            if _remaining() < 25.0:
                break
            res = _random_rank21_attempt()
            if res is not None:
                U21, V21, W21, r21 = res
                return _to_dict(U21, V21, W21)
=======
    # If rank-21 not found, try random restarts and also try dropping pairs of columns
    # and adding back optimized columns
    if _remaining() > 35.0:
        def _random_rank21_attempt():
            r = 21
            x0 = np.random.randn(nm * r + mp * r + pn * r) * 0.1

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 500, 'gtol': 1e-8, 'factr': 1e8})
                if res.fun < 1e-10:
                    x = res.x
                    U = x[:nm * r].reshape(nm, r)
                    V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W = x[nm * r + mp * r:].reshape(pn, r)
                    if _verify_full(U, V, W, n, m, p, tol=1e-6):
                        return U.tolist(), V.tolist(), W.tolist(), r
            except Exception:
                pass
            return None

        for _ in range(5):
            if _remaining() < 20.0:
                break
            res = _random_rank21_attempt()
            if res is not None:
                U21, V21, W21, r21 = res
                return _to_dict(U21, V21, W21)
>>>>>>> REPLACE
