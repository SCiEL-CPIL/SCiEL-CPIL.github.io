<<<<<<< SEARCH
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling
        # Use 10 initial points to find a good start
        n_global = 10
        candidates = []
        
        for i in range(n_global):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_used += 1
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if not candidates:
            return best_x, best_f

        # Sort and get best start
        candidates.sort(key=lambda t: t[0])
        x0 = candidates[0][1]
        
        # Phase 2: Main CMA-ES run
        remaining = total_budget - calls_used
        # Reserve some budget for polishing
        polish_budget = min(50, remaining // 10)
        main_budget = remaining - polish_budget
        
        cma_es(x0, 1.5, main_budget, rng)
        
        # Phase 3: Local polish around the best point
        if best_x is not None and not check_deadline():
            remaining_for_polish = total_budget - calls_used
            if remaining_for_polish > 5:
                local_polish(best_x, remaining_for_polish, rng)
        
        return best_x, best_f
=======
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling with stratified approach
        # Use Halton-like sequence for better coverage
        n_global = 15
        candidates = []
        
        # Stratified sampling: divide each dimension into bins
        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim."""
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(1, dim + 1):
                    f = 1.0
                    result = 0.0
                    i_copy = i
                    while i_copy > 0:
                        f /= d
                        result += f * (i_copy % d)
                        i_copy //= d
                    point.append(result)
                points.append(point)
            return np.array(points)
        
        # Generate stratified initial points
        halton_pts = halton_sequence(n_global, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        halton_pts = halton_pts * 10.0 - 5.0
        
        for i in range(n_global):
            if check_deadline():
                break
            x = halton_pts[i].copy()
            f = func(x)
            calls_used += 1
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if not candidates:
            return best_x, best_f

        # Sort and get best start
        candidates.sort(key=lambda t: t[0])
        x0 = candidates[0][1]
        
        # Phase 2: Main CMA-ES run
        remaining = total_budget - calls_used
        # Reserve budget for polishing: use 15% but at least 30 evals
        polish_budget = max(30, min(60, int(remaining * 0.15)))
        main_budget = remaining - polish_budget
        
        cma_es(x0, 1.5, main_budget, rng)
        
        # Phase 3: Local polish around the best point with adaptive sigma
        if best_x is not None and not check_deadline():
            remaining_for_polish = total_budget - calls_used
            if remaining_for_polish > 5:
                # Estimate current sigma from recent improvements
                # Use a smaller initial sigma for polishing
                local_polish(best_x, remaining_for_polish, rng)
        
        return best_x, best_f
>>>>>>> REPLACE

