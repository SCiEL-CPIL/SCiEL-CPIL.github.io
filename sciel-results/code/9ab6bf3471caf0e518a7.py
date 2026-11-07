        # Local Search Polish Phase
        if polish_budget > 0 and time.perf_counter() < deadline - 5:
            # Get non-dominated points from the final population
            final_fronts = fast_non_dominated_sort(F)
            if final_fronts:
                front_0 = final_fronts[0]
                # Select up to 5 best points (by crowding distance) for polishing
                cd_front = crowding_distance(F, front_0)
                # Sort by crowding distance descending (most diverse first)
                sorted_indices = sorted(front_0, key=lambda i: cd_front[front_0.index(i)], reverse=True)
                num_polish = min(5, len(sorted_indices))
                
                for idx in sorted_indices[:num_polish]:
                    if time.perf_counter() > deadline - 5 or evals >= budget:
                        break
                    
                    x_start = X[idx].copy()
                    f_start = F[idx].copy()
                    
                    # Define a scalarized objective to minimize.
                    # Minimize the sum of objectives to push towards the ideal point.
                    
                    def scalarized_objective(x, evals_counter=[0]):
                        fx = problem(x)
                        evals_counter[0] += 1
                        # We need to update the outer evals variable. 
                        # Python closure issue: evals is an int, not mutable.
                        # We can use a list or a class attribute, or just return the count.
                        # Let's use a nonlocal variable if possible, but we are in a method.
                        # We can't easily modify 'evals' from inside the nested function without 'nonlocal'.
                        # Let's use a helper list.
                        return np.sum(fx)
                    
                    # We need to count evaluations correctly.
                    # Let's wrap the problem call.
                    
                    def counting_problem(x):
                        nonlocal evals
                        fx = problem(x)
                        evals += 1
                        return fx

                    def scalarized_objective(x):
                        fx = counting_problem(x)
                        return np.sum(fx)
                    
                    # Use Nelder-Mead for local search
                    # Limit function calls
                    try:
                        res = minimize(scalarized_objective, x_start, method='Nelder-Mead', 
                                       options={'maxiter': 10, 'maxfev': 5, 'xatol': 1e-4, 'fatol': 1e-4})
                        
                        # Evaluate the result if it improved
                        if res.fun < np.sum(f_start):
                            x_polished = np.clip(res.x, lo, hi)
                            f_polished = problem(x_polished)
                            evals += 1
                            all_F_list.append(f_polished.copy())
                            
                    except Exception:
                        pass
