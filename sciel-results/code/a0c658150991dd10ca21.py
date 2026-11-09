    # Local search from the best candidate
    if best_c is not None:
        # Use approximate score for speed
        current_score = evaluate_approx(best_c)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(1, 23): # Flip any bit except c0
                c_trial = best_c.copy()
                c_trial[i] *= -1
                
                score_trial = evaluate_approx(c_trial)
                if score_trial < current_score - 1e-6:
                    best_c = c_trial
                    current_score = score_trial
                    improved = True
                    break # Restart loop
                    
        # Final refinement of the local search result
        final_score = evaluate_sup_norm(best_c)
        if final_score < best_score:
            best_score = final_score
