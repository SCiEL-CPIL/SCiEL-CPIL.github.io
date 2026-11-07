<<<<<<< SEARCH
            # Update best
            min_idx = int(np.argmin(fx_list))
            if fx_list[min_idx] < best_f:
                best_f = fx_list[min_idx]
                best_x = simplex[min_idx].copy()

            # Check for convergence or stagnation
            spread = np.max(fx_list) - np.min(fx_list)
            if spread < 1e-12:
                break
=======
            # Update best
            min_idx = int(np.argmin(fx_list))
            if fx_list[min_idx] < best_f:
                best_f = fx_list[min_idx]
                best_x = simplex[min_idx].copy()

            # Check for convergence or stagnation
            # Calculate geometric spread of the simplex
            max_dist = 0
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    d = np.linalg.norm(simplex[i] - simplex[j])
                    if d > max_dist:
                        max_dist = d
            
            if max_dist < 1e-5:
                # Simplex has collapsed, restart around best point
                # Only if we have enough budget left
                if calls + dim < budget:
                    center = best_x.copy()
                    simplex = [center.copy()]
                    fx_list = [best_f]
                    restart_step = 1.0
                    for i in range(dim):
                        v = center.copy()
                        # Choose direction that stays in bounds if possible, or just clip
                        # Simple approach: try +step, if out of bounds, try -step
                        if center[i] + restart_step <= 5.0:
                            v[i] = center[i] + restart_step
                        else:
                            v[i] = center[i] - restart_step
                        simplex.append(v)
                        fx_list.append(func(v))
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break
            elif max_dist < 1e-12:
                break
>>>>>>> REPLACE
