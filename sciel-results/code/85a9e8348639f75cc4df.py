<<<<<<< SEARCH
        # Simulated annealing
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 15:
            input_c = best_circles if best_circles is not None else resolved
            sa_circles, sa_score = _simulated_annealing(
                input_c, max_iter=40000, t_init=0.04, t_min=0.0005,
                time_limit=min(remaining * 0.3, 20), seed=hash(name) % 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
=======
        # Simulated annealing
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 15:
            input_c = best_circles if best_circles is not None else resolved
            sa_circles, sa_score = _simulated_annealing(
                input_c, max_iter=30000, t_init=0.04, t_min=0.0005,
                time_limit=min(remaining * 0.3, 15), seed=hash(name) % 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
>>>>>>> REPLACE
