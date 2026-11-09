        remaining = budget - calls_made
        if remaining >= 10:
            max_iter = max(1, remaining // 8)
            try:
                result = minimize(counted_func, best_x, method='Nelder-Mead', 
                                  options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass
        elif remaining > 0:
            # Do a few manual coordinate steps to use remaining budget safely
            for i in range(remaining):
                if time.time() > deadline:
                    break
                # Try small perturbations
                for d in range(dim):
                    if calls_made >= budget: break
                    x_new = best_x.copy()
                    x_new[d] += 0.1
                    if x_new[d] > 5: x_new[d] = 5
                    if x_new[d] < -5: x_new[d] = -5
                    f_new = counted_func(x_new)
                    if f_new < best_f:
                        best_x = x_new
                        best_f = f_new
