        # Propose a move: flip 1 random position
        flip_idx = np.random.randint(N)
        old_val = current_step[flip_idx]
        new_val = -old_val
        
        d = -2.0 * old_val
        shift1 = 2 * flip_idx
        shift2 = shift1 + 1
        
        # Calculate cross term
        # Using np.roll
        cross_term = d * (np.roll(f_rev, shift1) + np.roll(f_rev, shift2)) * dx
        
        new_autoconv = autoconv + 2.0 * cross_term + H_delta
        new_c3 = np.max(np.abs(new_autoconv))
        
        delta = new_c3 - current_c3
        
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_step[flip_idx] = new_val
            f_fine[shift1] = new_val
            f_fine[shift2] = new_val
            f_rev[(-shift1) % M] = new_val
            f_rev[(-shift2) % M] = new_val
            autoconv = new_autoconv
            current_c3 = new_c3
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        
        T *= cooling_rate
