<<<<<<< SEARCH
            # Evaluate initial
            f0 = obj(x0)
            
            # Differential evolution
            pop_size = min(30, max(10, n_layers * 2))
            de_time_limit = time_budget * 0.6
            de_start = time.time()
            
            best_x, best_f = _differential_evolution(
                obj, bounds, pop_size=pop_size, maxiter=50,
                F=0.8, CR=0.9, seed=42,
                time_limit=de_time_limit, start_time=de_start
            )
            
            # Polish with Nelder-Mead
            polish_time_limit = time_budget * 0.4
            polish_start = time.time()
            
            best_x, best_f = _polish(best_x, obj, bounds,
                                     time_limit=polish_time_limit, start_time=polish_start)
=======
            # Evaluate initial
            f0 = obj(x0)
            
            # Particle Swarm Optimization
            swarm_size = min(40, max(15, n_layers * 3))
            pso_time_limit = time_budget * 0.6
            pso_start = time.time()
            
            best_x, best_f = _particle_swarm_optimization(
                obj, bounds, swarm_size=swarm_size, maxiter=80,
                seed=42, time_limit=pso_time_limit, start_time=pso_start
            )
            
            # Polish with SciPy Nelder-Mead
            polish_time_limit = time_budget * 0.4
            polish_start = time.time()
            
            best_x, best_f = _polish(best_x, obj, bounds,
                                     time_limit=polish_time_limit, start_time=polish_start)
>>>>>>> REPLACE
