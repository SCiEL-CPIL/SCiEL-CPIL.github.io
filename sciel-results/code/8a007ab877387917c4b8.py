<<<<<<< SEARCH
    # Phase 3: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-10)

        # Final high-precision polish
        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=300, ftol=1e-12)
=======
    # Phase 3: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 8:
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-10)

        # Second SLSQP with different beta
        if time.time() < _DEADLINE - 5:
            _run_optimization(best_h, beta=150.0, maxiter=200, ftol=1e-11)

        # Final high-precision polish
        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=300.0, maxiter=300, ftol=1e-12)
>>>>>>> REPLACE
