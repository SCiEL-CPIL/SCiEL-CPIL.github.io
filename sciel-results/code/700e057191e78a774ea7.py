<<<<<<< SEARCH
        # Acceptance Criterion
        if c_iter < best_cost:
            best_cost = c_iter
            best_p = p_iter.copy()
            current_p = p_iter.copy()
            current_cost = c_iter
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            # Simulated Annealing acceptance
            delta = c_iter - current_cost
            if delta < 0:
                # Better than current, accept
                current_p = p_iter.copy()
                current_cost = c_iter
            else:
                # Worse, accept with probability
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_p = p_iter.copy()
                        current_cost = c_iter
                # Else reject (keep current)
        
        T *= cooling_rate
=======
        # Acceptance Criterion with restart mechanism
        if c_iter < best_cost:
            best_cost = c_iter
            best_p = p_iter.copy()
            current_p = p_iter.copy()
            current_cost = c_iter
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            # Simulated Annealing acceptance
            delta = c_iter - current_cost
            if delta < 0:
                # Better than current, accept
                current_p = p_iter.copy()
                current_cost = c_iter
            else:
                # Worse, accept with probability
                if T > min_T:
                    # Use scaled delta to avoid overflow for large gaps
                    scaled_delta = min(delta, 50 * T)
                    prob = math.exp(-scaled_delta / T)
                    if random.random() < prob:
                        current_p = p_iter.copy()
                        current_cost = c_iter
                # Else reject (keep current)
        
        # Periodic restart: if stuck too long, reset to best and increase perturbation
        if consecutive_no_improve > 30 and iteration % 10 == 0:
            current_p = best_p.copy()
            current_cost = best_cost
            consecutive_no_improve = 0
        
        T = max(T * cooling_rate, min_T)
>>>>>>> REPLACE
