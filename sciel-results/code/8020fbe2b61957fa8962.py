<<<<<<< SEARCH
        # Final local refinement around the best point found using Nelder-Mead
        if best_x is not None and total_calls < budget - 5 and time.time() < deadline - 10:
            nm_budget = budget - total_calls - 5
            if nm_budget > 10:
                best_x, best_f = self._nelder_mead(func, best_x, best_f, dim, nm_budget, deadline)
        
        if best_x is None:
            # Fallback: should not happen, but be safe
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
        
        return best_x, best_f
=======
        # Final local refinement around the best point found using Nelder-Mead
        if best_x is not None and total_calls < budget - 5 and time.time() < deadline - 10:
            nm_budget = budget - total_calls - 5
            if nm_budget > 10:
                best_x, best_f = self._nelder_mead(func, best_x, best_f, dim, nm_budget, deadline)
        
        # Additional high-precision polish if we have budget remaining
        if best_x is not None and total_calls < budget - 3 and time.time() < deadline - 5:
            # Do a few more fine-grained evaluations around the best point
            fine_sigma = 0.01
            for _ in range(min(10, budget - total_calls - 3)):
                if time.time() >= deadline - 3:
                    break
                y = np.clip(best_x + fine_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
        
        if best_x is None:
            # Fallback: should not happen, but be safe
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
        
        return best_x, best_f
>>>>>>> REPLACE
