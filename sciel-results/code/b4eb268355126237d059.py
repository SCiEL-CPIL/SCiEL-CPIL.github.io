            # Round back to signs
            if result.success or result.fun < best_obj:
                new_signs = np.where(result.x < np.pi / 2, 1, -1).astype(np.int8)
                new_signs = np.atleast_1d(new_signs)
                
                # Check connectivity
                if is_connected_lift(new_signs):
                    new_obj = objective(new_signs)
                    if new_obj < best_obj - 1e-12:
                        best_obj = new_obj
                        best_signs = new_signs.copy()
                        # Polish with discrete local search
                        try_improve(best_signs.copy())
