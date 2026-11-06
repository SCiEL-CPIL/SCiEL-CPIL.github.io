<<<<<<< SEARCH
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    
    num_restarts = 5
    iters_per_restart = 8000
    T_init = 0.3
    T_final = 0.0005
    
    for restart in range(num_restarts):
        # Check time
        if time.time() - start_time > time_budget - 10:
            break
        
        # Initialize current state
        if restart == 0:
            current_seq = list(best_seq)
            current_scale = best_scale
            current_score = best_score
        else:
            # Random restart
            n_layers = np.random.randint(2, max_layers + 1)
            current_seq = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
            current_scale = np.random.uniform(0.5, 2.0)
            current_score = _evaluate(current_seq, current_scale, problem)
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq)
                best_scale = current_scale
        
        cooling = (T_final / T_init) ** (1.0 / iters_per_restart)
        
        for i in range(iters_per_restart):
            # Check time
            if time.time() - start_time > time_budget - 5:
                break
            
            T = T_init * (cooling ** i)
            
            # Generate a neighbor
            r = np.random.rand()
            if r < 0.35:
                # Change a random layer
                new_seq = list(current_seq)
                if len(new_seq) > 0:
                    idx = np.random.randint(len(new_seq))
                    new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
                else:
                    new_seq = ['L']
                new_scale = current_scale
            elif r < 0.55:
                # Add a layer
                if len(current_seq) < max_layers:
                    new_seq = list(current_seq)
                    pos = np.random.randint(len(new_seq) + 1)
                    new_layer = 'L' if np.random.rand() < 0.5 else 'H'
                    new_seq.insert(pos, new_layer)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.75:
                # Remove a layer
                if len(current_seq) > 1:
                    new_seq = list(current_seq)
                    idx = np.random.randint(len(new_seq))
                    new_seq.pop(idx)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.90:
                # Perturb the scale
                new_seq = list(current_seq)
                new_scale = current_scale * (1.0 + 0.15 * np.random.randn())
                new_scale = max(0.3, min(3.0, new_scale))
            else:
                # Swap two adjacent layers
                new_seq = list(current_seq)
                if len(new_seq) >= 2:
                    idx = np.random.randint(len(new_seq) - 1)
                    new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
                new_scale = current_scale
            
            # Evaluate
            new_score = _evaluate(new_seq, new_scale, problem)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score:
                current_seq = new_seq
                current_scale = new_scale
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = list(current_seq)
                    best_scale = current_scale
            else:
                delta = new_score - current_score
                if delta > 0 and T > 0:
                    accept_prob = np.exp(-delta / T)
                    if np.random.rand() < accept_prob:
                        current_seq = new_seq
                        current_scale = new_scale
                        current_score = new_score
    
    # Final refinement: optimize scale for best sequence
    best_scale, best_score = _optimize_scale_for_seq(best_seq, problem)
    
    # Try small modifications to best sequence
    if time.time() - start_time < time_budget - 3:
        for _ in range(200):
            trial_seq = list(best_seq)
            r = np.random.rand()
            if r < 0.4 and len(trial_seq) < max_layers:
                pos = np.random.randint(len(trial_seq) + 1)
                trial_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
            elif r < 0.7 and len(trial_seq) > 1:
                idx = np.random.randint(len(trial_seq))
                trial_seq.pop(idx)
            else:
                if len(trial_seq) > 0:
                    idx = np.random.randint(len(trial_seq))
                    trial_seq[idx] = 'H' if trial_seq[idx] == 'L' else 'L'
            
            s, score = _optimize_scale_for_seq(trial_seq, problem)
            if score < best_score:
                best_score = score
                best_seq = list(trial_seq)
                best_scale = s
=======
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    
    num_restarts = 8
    iters_per_restart = 12000
    T_init = 0.5
    T_final = 0.001
    
    for restart in range(num_restarts):
        # Check time
        if time.time() - start_time > time_budget - 10:
            break
        
        # Initialize current state
        if restart == 0:
            current_seq = list(best_seq)
            current_scale = best_scale
            current_score = best_score
        else:
            # Random restart with bias toward good layer counts
            n_layers = np.random.choice([3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20], 
                                        p=[0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02])
            current_seq = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
            current_scale = np.random.uniform(0.5, 2.5)
            current_score = _evaluate(current_seq, current_scale, problem)
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq)
                best_scale = current_scale
        
        cooling = (T_final / T_init) ** (1.0 / iters_per_restart)
        
        for i in range(iters_per_restart):
            # Check time
            if time.time() - start_time > time_budget - 5:
                break
            
            T = T_init * (cooling ** i)
            
            # Generate a neighbor
            r = np.random.rand()
            if r < 0.40:
                # Change a random layer
                new_seq = list(current_seq)
                if len(new_seq) > 0:
                    idx = np.random.randint(len(new_seq))
                    new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
                else:
                    new_seq = ['L']
                new_scale = current_scale
            elif r < 0.55:
                # Add a layer
                if len(current_seq) < max_layers:
                    new_seq = list(current_seq)
                    pos = np.random.randint(len(new_seq) + 1)
                    new_layer = 'L' if np.random.rand() < 0.5 else 'H'
                    new_seq.insert(pos, new_layer)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.70:
                # Remove a layer
                if len(current_seq) > 1:
                    new_seq = list(current_seq)
                    idx = np.random.randint(len(new_seq))
                    new_seq.pop(idx)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.85:
                # Perturb the scale
                new_seq = list(current_seq)
                new_scale = current_scale * (1.0 + 0.10 * np.random.randn())
                new_scale = max(0.3, min(3.0, new_scale))
            else:
                # Swap two adjacent layers
                new_seq = list(current_seq)
                if len(new_seq) >= 2:
                    idx = np.random.randint(len(new_seq) - 1)
                    new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
                new_scale = current_scale
            
            # Evaluate
            new_score = _evaluate(new_seq, new_scale, problem)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score:
                current_seq = new_seq
                current_scale = new_scale
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = list(current_seq)
                    best_scale = current_scale
            else:
                delta = new_score - current_score
                if delta > 0 and T > 0:
                    accept_prob = np.exp(-delta / T)
                    if np.random.rand() < accept_prob:
                        current_seq = new_seq
                        current_scale = new_scale
                        current_score = new_score
    
    # Final refinement: optimize scale for best sequence
    best_scale, best_score = _optimize_scale_for_seq(best_seq, problem)
    
    # Try small modifications to best sequence
    if time.time() - start_time < time_budget - 3:
        for _ in range(300):
            trial_seq = list(best_seq)
            r = np.random.rand()
            if r < 0.4 and len(trial_seq) < max_layers:
                pos = np.random.randint(len(trial_seq) + 1)
                trial_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
            elif r < 0.7 and len(trial_seq) > 1:
                idx = np.random.randint(len(trial_seq))
                trial_seq.pop(idx)
            else:
                if len(trial_seq) > 0:
                    idx = np.random.randint(len(trial_seq))
                    trial_seq[idx] = 'H' if trial_seq[idx] == 'L' else 'L'
            
            s, score = _optimize_scale_for_seq(trial_seq, problem)
            if score < best_score:
                best_score = score
                best_seq = list(trial_seq)
                best_scale = s
>>>>>>> REPLACE

