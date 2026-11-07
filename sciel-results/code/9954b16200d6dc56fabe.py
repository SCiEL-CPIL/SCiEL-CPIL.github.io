<<<<<<< SEARCH
    # To make local search faster, we can compute the "contribution" of each element to the total score.
    # contrib[i] = sum of comp[fam[i]][fam[j]] for j != i.
    # Total score = 0.5 * sum(contrib[i]).
    # If we swap out i and in new_mask:
    # New score = (Total - contrib[i]) + sum(comp[new_mask][fam[j]] for j != i)
    # Wait, Total = 0.5 * sum(contrib).
    # Let S = sum(contrib). Total = S / 2.
    # Removing i: S' = S - contrib[i].
    # Adding new: S'' = S' + new_contrib, where new_contrib = sum(comp[new_mask][fam[j]] for j in fam, j!=i).
    # Delta Total = 0.5 * (S'' - S) = 0.5 * (new_contrib - contrib[i]).
    # So we just need to maximize (new_contrib - contrib[i]).
    
    while time.time() < deadline:
        improved = False
        
        # Compute contribs for current family
        contribs = [0] * m
        for i in range(m):
            c = 0
            fi = best_family[i]
            for j in range(m):
                if i != j:
                    c += comp[fi][best_family[j]]
            contribs[i] = c
            
        # Try to swap each i with some best candidate
        # We can precompute for each candidate s the sum of comp[s][fam[j]]?
        # That is 4096 * 40 = 160k ops.
        # Then for each i, we look for max (sum - comp[s][fam[i]] - contribs[i]).
        # Note: sum_comp_s = sum(comp[s][fam[j]] for all j).
        # new_contrib = sum_comp_s - comp[s][fam[i]] (since s is not in family, comp[s][s] is 0 anyway, but s != fam[i] is required).
        
        # Precompute sum_comp for all s not in family
        fam_set = set(best_family)
        sum_comps = {}
        for s in range(total_masks):
            if s in fam_set:
                continue
            sc = 0
            for f in best_family:
                sc += comp[s][f]
            sum_comps[s] = sc
            
        # Now find best swap
        best_delta = 0
        best_i = -1
        best_s = -1
        
        for i in range(m):
            fi = best_family[i]
            # We want to maximize (sum_comp_s - comp[s][fi] - contribs[i])
            # Which is sum_comp_s - comp[s][fi]
            # Since contribs[i] is constant for this i.
            max_val = -1
            best_s_for_i = -1
            
            for s, sc in sum_comps.items():
                val = sc - comp[s][fi]
                if val > max_val:
                    max_val = val
                    best_s_for_i = s
            
            delta = max_val - contribs[i]
            if delta > best_delta:
                best_delta = delta
                best_i = i
                best_s = best_s_for_i
                
        if best_delta > 0 and best_i != -1 and best_s != -1:
            # Perform swap
            old_val = best_family[best_i]
            best_family[best_i] = best_s
            improved = True
            # Update used set? Not strictly necessary if we use fam_set derived from best_family each time.
            # But for the next iteration's precomputation, we just rebuild fam_set.
            
        if not improved:
            break
=======
    # To make local search faster, we can compute the "contribution" of each element to the total score.
    # contrib[i] = sum of comp[fam[i]][fam[j]] for j != i.
    # Total score = 0.5 * sum(contrib[i]).
    # If we swap out i and in new_mask:
    # New score = (Total - contrib[i]) + sum(comp[new_mask][fam[j]] for j != i)
    # Wait, Total = 0.5 * sum(contrib).
    # Let S = sum(contrib). Total = S / 2.
    # Removing i: S' = S - contrib[i].
    # Adding new: S'' = S' + new_contrib, where new_contrib = sum(comp[new_mask][fam[j]] for j in fam, j!=i).
    # Delta Total = 0.5 * (S'' - S) = 0.5 * (new_contrib - contrib[i]).
    # So we just need to maximize (new_contrib - contrib[i]).
    
    # Simulated annealing parameters
    best_score = current_score
    best_family_copy = best_family[:]
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.9995
    
    T = T_start
    no_improve_count = 0
    
    while time.time() < deadline:
        # Compute contribs for current family
        contribs = [0] * m
        for i in range(m):
            c = 0
            fi = best_family[i]
            for j in range(m):
                if i != j:
                    c += comp[fi][best_family[j]]
            contribs[i] = c
            
        # Precompute sum_comp for all s not in family
        fam_set = set(best_family)
        sum_comps = {}
        for s in range(total_masks):
            if s in fam_set:
                continue
            sc = 0
            for f in best_family:
                sc += comp[s][f]
            sum_comps[s] = sc
            
        # Find best swap (for greedy improvement)
        best_delta = 0
        best_i = -1
        best_s = -1
        
        for i in range(m):
            fi = best_family[i]
            max_val = -1
            best_s_for_i = -1
            
            for s, sc in sum_comps.items():
                val = sc - comp[s][fi]
                if val > max_val:
                    max_val = val
                    best_s_for_i = s
            
            delta = max_val - contribs[i]
            if delta > best_delta:
                best_delta = delta
                best_i = i
                best_s = best_s_for_i
        
        # Decide whether to accept the best move or a random move (annealing)
        accept_move = False
        if best_delta > 0:
            accept_move = True
            move_i = best_i
            move_s = best_s
            move_delta = best_delta
        else:
            # Accept a random move with probability based on temperature
            if random.random() < 0.3:  # 30% chance to try a random move when no improving move exists
                # Pick a random i and a random s
                move_i = random.randint(0, m - 1)
                fi = best_family[move_i]
                # Pick a random s from sum_comps
                if sum_comps:
                    s_list = list(sum_comps.keys())
                    move_s = s_list[random.randint(0, len(s_list) - 1)]
                    # Compute delta for this move
                    new_contrib = sum_comps[move_s] - comp[move_s][fi]
                    move_delta = new_contrib - contribs[move_i]
                    
                    # Accept with Metropolis criterion
                    if move_delta >= 0 or random.random() < pow(2.718281828, move_delta / T):
                        accept_move = True
                    else:
                        move_delta = None
                else:
                    move_delta = None
            else:
                move_delta = None
        
        if accept_move and move_delta is not None:
            old_val = best_family[move_i]
            best_family[move_i] = move_s
            
            # Update score
            current_score = best_score if move_delta == best_delta and move_delta > 0 else current_score
            # Actually, let's recompute or update properly
            # The delta is 0.5 * (new_contrib - contrib[i]) for the total score
            # Wait, in the code above, delta = max_val - contribs[i] where max_val is the new_contrib
            # And the actual score change is 0.5 * delta
            # Let me re-derive:
            # Total = 0.5 * S where S = sum(contribs)
            # When we swap i with s:
            # S_new = S - contribs[i] + new_contrib
            # Total_new = 0.5 * S_new
            # Delta Total = 0.5 * (new_contrib - contribs[i]) = 0.5 * delta
            # So current_score should change by 0.5 * move_delta
            
            current_score += 0.5 * move_delta
            
            if current_score > best_score:
                best_score = current_score
                best_family_copy = best_family[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        else:
            no_improve_count += 1
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
            # If we've had many non-improving steps and temperature is low, restart
            if no_improve_count > 100 and T < 0.5:
                # Restart from best known
                best_family = best_family_copy[:]
                current_score = best_score
                T = T_start * 0.5
                no_improve_count = 0
>>>>>>> REPLACE

