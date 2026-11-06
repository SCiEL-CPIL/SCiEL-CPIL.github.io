<<<<<<< SEARCH
    # Phase 2: Intensive stochastic search with simulated annealing
    if best_cell is not None and time.time() < DEADLINE - 40:
        print(f"Phase 2: Stochastic search with annealing...")
        current = best_cell.copy()
        current_score = best_score
        T = max(0.05, abs(current_score) * 0.3)
        T_min = 0.01
        cooling = 0.995

        for iteration in range(2000):
            if time.time() > DEADLINE - 20:
                break
            mutated = mutate_cell(current, n_mutations=random.randint(1, 3))
            fast_score = evaluate_cell_fast(mutated)
            
            # Simulated annealing acceptance
            if fast_score > current_score:
                accept = True
            else:
                delta = current_score - fast_score
                accept = random.random() < np.exp(-delta / max(T, 1e-10))
            
            if accept:
                full_score = evaluate_cell(mutated)
                if full_score > current_score:
                    current = mutated
                    current_score = full_score
                    update_best(current, current_score)
                else:
                    # Keep current but still update T
                    pass
            T *= cooling
            T = max(T, T_min)
=======
    # Phase 2: Local refinement around best Moiré parameters
    # Instead of random pixel flips, we refine the continuous parameters (f1, f2, p1, p2, thr)
    # This keeps the structure physically meaningful and symmetric.
    
    if best_cell is not None and time.time() < DEADLINE - 40:
        print(f"Phase 2: Parameter refinement...")
        
        # We need to recover the parameters that generated best_cell? 
        # Or just do a local search in parameter space around a few good candidates?
        # Since we don't store the best params in `best_cell`, let's use the top few from Phase 1.
        
        # Let's take the top 5 from fast_scores and refine them.
        top_candidates = fast_scores[:5]
        
        for score, cell, params in top_candidates:
            if time.time() > DEADLINE - 20:
                break
            
            f1, f2, p1, p2, thr = params
            
            # Define a small neighborhood in parameter space
            # Step sizes
            df = 0.1
            dp = np.pi / 16
            dthr = 0.05
            
            current_best_params = params
            current_best_score = score # Use fast score for comparison in this phase initially? 
            # Actually, let's use the full score of the initial candidate
            current_best_score = evaluate_cell(cell)
            
            # Simple hill climbing / local search
            for step in range(20):
                if time.time() > DEADLINE - 10:
                    break
                
                improved = False
                # Try perturbations
                for df1 in [-df, df]:
                    for df2 in [-df, df]:
                        for dp1 in [-dp, dp]:
                            for dp2 in [-dp, dp]:
                                for dthr in [-dthr, dthr]:
                                    if time.time() > DEADLINE - 10:
                                        break
                                    
                                    new_f1 = f1 + df1
                                    new_f2 = f2 + df2
                                    new_p1 = p1 + dp1
                                    new_p2 = p2 + dp2
                                    new_thr = thr + dthr
                                    
                                    # Bounds
                                    if new_f1 < 1.0 or new_f1 > 6.0: continue
                                    if new_f2 < 1.0 or new_f2 > 6.0: continue
                                    if new_thr < 0.1 or new_thr > 1.0: continue
                                    
                                    new_cell = generate_moire_vectorized(new_f1, new_f2, new_p1, new_p2, new_thr)
                                    new_score = evaluate_cell_fast(new_cell)
                                    
                                    if new_score > current_best_score * 0.98: # Threshold to avoid noise
                                        full_new_score = evaluate_cell(new_cell)
                                        if full_new_score > current_best_score:
                                            current_best_score = full_new_score
                                            current_best_params = (new_f1, new_f2, new_p1, new_p2, new_thr)
                                            update_best(new_cell, full_new_score)
                                            improved = True
                                            # Update base params for next iteration? 
                                            # Or keep exploring from original? 
                                            # Let's keep exploring from the best found so far (hill climbing)
                                            f1, f2, p1, p2, thr = current_best_params
                                            break
            print(f"  Refined candidate to score: {current_best_score:.4f}")
>>>>>>> REPLACE

