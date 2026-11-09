<<<<<<< SEARCH
        try:
            es = CMAEvolutionStrategy(cma_start, cma_sigma, cma_opts)
            while not es.stop() and has_budget():
                solutions = es.ask()
                for sol in solutions:
                    if not has_budget():
                        break
                    counted_func(np.array(sol))
                es.tell(solutions, [counted_func(np.array(s)) for s in solutions] if False else None)
                # Actually, we need to collect fitness values properly
                break  # Fallback to simpler loop below if the above doesn't work cleanly
        except Exception:
            pass
        
        # CMA-ES loop that properly tracks fitness
=======
        # CMA-ES loop that properly tracks fitness
>>>>>>> REPLACE
