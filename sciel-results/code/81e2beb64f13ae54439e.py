<<<<<<< SEARCH
    # Phase 5: Local perturbation + re-optimize to escape local minima
    if best_X is not None and time.time() - start_time < deadline:
        time_remaining = deadline - (time.time() - start_time)
        for _ in range(3):
            if time.time() - start_time > deadline:
                break
            X_opt, E_opt = _local_perturb_restart(best_X, n_perturb=5, step_size=0.08,
                                                   time_limit=time_remaining * 0.2, rng=rng)
            if E_opt < best_energy:
                best_energy = E_opt
                best_X = X_opt
            # Also try L-BFGS on the perturbed result
            if time.time() - start_time < deadline:
                time_remaining = deadline - (time.time() - start_time)
                X_opt2, E_opt2 = _lbfgs_polish(X_opt, time_limit=time_remaining * 0.3, maxiter=50)
                if E_opt2 < best_energy:
                    best_energy = E_opt2
                    best_X = X_opt2
=======
    # Phase 5: Local perturbation + re-optimize to escape local minima
    if best_X is not None and time.time() - start_time < deadline:
        time_remaining = deadline - (time.time() - start_time)
        for _ in range(3):
            if time.time() - start_time > deadline:
                break
            X_opt, E_opt = _local_perturb_restart(best_X, n_perturb=5, step_size=0.08,
                                                   time_limit=time_remaining * 0.2, rng=rng)
            if E_opt < best_energy:
                best_energy = E_opt
                best_X = X_opt
            # Also try L-BFGS on the perturbed result
            if time.time() - start_time < deadline:
                time_remaining = deadline - (time.time() - start_time)
                X_opt2, E_opt2 = _lbfgs_polish(X_opt, time_limit=time_remaining * 0.3, maxiter=50)
                if E_opt2 < best_energy:
                    best_energy = E_opt2
                    best_X = X_opt2
            # Try another round of parallel tempering from improved state
            if best_X is not None and time.time() - start_time < deadline:
                time_remaining = deadline - (time.time() - start_time)
                X_opt3, E_opt3 = _parallel_tempering(best_X, max_iters=500, T_init=0.01, T_min=1e-6,
                                                      T_decay=0.998, n_replicas=3,
                                                      time_limit=time_remaining * 0.2, rng=rng)
                if E_opt3 < best_energy:
                    best_energy = E_opt3
                    best_X = X_opt3
>>>>>>> REPLACE
