<<<<<<< SEARCH
    # Phase 2: Local polish with Nelder-Mead (more robust than gradient methods here)
    if time.time() < deadline - 20.0:
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res2.fun > best_score:
                best_params = res2.x
                best_score = -res2.fun
        except Exception:
            pass
=======
    # Phase 2: Local polish with Nelder-Mead (more robust than gradient methods here)
    if time.time() < deadline - 20.0:
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res2.fun > best_score:
                best_params = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    # Phase 2b: Smooth surrogate polish with L-BFGS-B for finer convergence
    if time.time() < deadline - 15.0:
        try:
            # Use soft-min objective with moderate beta for smooth gradients
            res2b = minimize(
                lambda p: _objective_soft(p, beta=100.0),
                best_params,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if -res2b.fun > best_score:
                best_params = res2b.x
                best_score = -res2b.fun
        except Exception:
            pass

    # Phase 2c: Final high-precision polish with very tight tolerances
    if time.time() < deadline - 10.0:
        try:
            res2c = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            if -res2c.fun > best_score:
                best_params = res2c.x
                best_score = -res2c.fun
        except Exception:
            pass
>>>>>>> REPLACE

