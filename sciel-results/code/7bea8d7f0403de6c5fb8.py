<<<<<<< SEARCH
        # Try multiple initial angles with different betas
        init_angles = list(range(0, 360, 15))
        beta_values = [20.0, 40.0, 80.0]
        
        for beta in beta_values:
            for init_angle in init_angles:
                if time.time() > time_budget:
                    break
                
                init_params = [cx_avg, cy_avg, float(init_angle)]
                
                try:
                    res = minimize(lambda p, b=beta: objective_outer_smooth(p, b), init_params, 
                                  method='Nelder-Mead',
                                  options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 1500})
                    
                    res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 4000})
                    
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_result = res2.x
                except Exception:
                    pass
            if time.time() > time_budget:
                break
=======
        # Try multiple initial angles with different betas
        init_angles = list(range(0, 360, 15))
        beta_values = [20.0, 40.0, 80.0, 150.0]
        
        for beta in beta_values:
            for init_angle in init_angles:
                if time.time() > time_budget:
                    break
                
                init_params = [cx_avg, cy_avg, float(init_angle)]
                
                try:
                    res = minimize(lambda p, b=beta: objective_outer_smooth(p, b), init_params, 
                                  method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 2000})
                    
                    res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 6000})
                    
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_result = res2.x
                except Exception:
                    pass
            if time.time() > time_budget:
                break
>>>>>>> REPLACE

