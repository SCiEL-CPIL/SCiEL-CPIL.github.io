    # Simulated Annealing
    T = 2.0
    cooling = 0.999
    no_improve_count = 0
    last_best_score = best_score
    
    while time.time() < deadline:
        # ... move logic ...
        
        if best_score < last_best_score:
            last_best_score = best_score
            no_improve_count = 0
            T = 2.0 # Restart cooling
        else:
            no_improve_count += 1
            if no_improve_count > 1000:
                # Reset to best known and restart
                adj = best_adj[:]
                current_score = best_score
                T = 2.0
                no_improve_count = 0
                last_best_score = best_score

        T *= cooling
        if T < 0.01:
            T = 0.01
