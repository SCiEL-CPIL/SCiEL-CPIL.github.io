        for i in range(n_restarts):
            if time.time() > deadline:
                break
                
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            # 1 call used
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.5
            
            # Loop for remaining budget of this run
            for _ in range(budgets[i] - 1):
                if time.time() > deadline:
                    return best_x, best_f
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
