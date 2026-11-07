<<<<<<< SEARCH
        # Phase 4: Final high-precision polish on best
        if calls_made < budget and remaining_time() > 5.0:
            local_budget = min(budget - calls_made, 100)
            local_calls = [0]
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 10.0
                return evaluate(x)
            
            try:
                # First pass: Nelder-Mead
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': min(local_budget, 60), 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass
            
            # Second pass: tighter Nelder-Mead if time/calls remain
            if calls_made < budget and remaining_time() > 3.0:
                remaining_local = min(budget - calls_made, 40)
                if remaining_local > 5:
                    local_calls2 = [0]
                    
                    def local_func2(x):
                        local_calls2[0] += 1
                        if local_calls2[0] > remaining_local:
                            return best_f + 10.0
                        return evaluate(x)
                    
                    try:
                        res2 = minimize(local_func2, best_x, method='Nelder-Mead',
                                     options={'maxiter': remaining_local, 'xatol': 1e-10, 'fatol': 1e-12})
                        if res2.fun < best_f:
                            best_f = float(res2.fun)
                            best_x = clip(res2.x)
                    except Exception:
                        pass

        return best_x, best_f
=======
        # Phase 4: Final high-precision polish on best
        if calls_made < budget and remaining_time() > 5.0:
            local_budget = min(budget - calls_made, 120)
            local_calls = [0]
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 10.0
                return evaluate(x)
            
            try:
                # First pass: Nelder-Mead with tighter tolerances
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': min(local_budget, 70), 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass
            
            # Second pass: even tighter Nelder-Mead if time/calls remain
            if calls_made < budget and remaining_time() > 3.0:
                remaining_local = min(budget - calls_made, 50)
                if remaining_local > 5:
                    local_calls2 = [0]
                    
                    def local_func2(x):
                        local_calls2[0] += 1
                        if local_calls2[0] > remaining_local:
                            return best_f + 10.0
                        return evaluate(x)
                    
                    try:
                        res2 = minimize(local_func2, best_x, method='Nelder-Mead',
                                     options={'maxiter': remaining_local, 'xatol': 1e-12, 'fatol': 1e-14,
                                              'adaptive': True})
                        if res2.fun < best_f:
                            best_f = float(res2.fun)
                            best_x = clip(res2.x)
                    except Exception:
                        pass
            
            # Third pass: coordinate descent as final polish
            if calls_made < budget and remaining_time() > 2.0:
                coord_budget = min(budget - calls_made, 30)
                step = 0.01
                for _ in range(coord_budget):
                    if calls_made >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        for direction in [1, -1]:
                            x_try = best_x.copy()
                            x_try[d] += direction * step
                            x_try = clip(x_try)
                            f_try = evaluate(x_try)
                            if f_try < best_f:
                                best_f = f_try
                                best_x = x_try.copy()
                                improved = True
                    if not improved:
                        step *= 0.5
                        if step < 1e-6:
                            break

        return best_x, best_f
>>>>>>> REPLACE
