<<<<<<< SEARCH
            # Delta for signs
            delta = -2.0 * signs[k]
            
            # Change in J = sum S_P^2
            delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
            
            # Combined objective: primarily sum-of-squares, with soft-max penalty
            # The soft-max term encourages reducing the worst AP sums
            idxs = pos_to_aps_arr[k]
            
            # Estimate change in soft-max by looking at the affected APs
            # For efficiency, only compute full soft-max change every 10 steps
            if steps % 10 == 0:
                old_soft = soft_max_obj(sums)
                # Compute new sums for affected APs only (approximation)
                new_sums_approx = sums.copy()
                new_sums_approx[idxs] += delta
                new_soft = soft_max_obj(new_sums_approx)
                delta_soft = new_soft - old_soft
                # Scale soft-max change to be comparable to delta_J
                delta_combined = delta_J + 5.0 * delta_soft
            else:
                delta_combined = delta_J
            
            # Metropolis Criterion
            accept = False
            if delta_combined < 0:
                accept = True
            else:
                if delta_combined / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_combined / T)
                    
            if accept:
                # Apply move
                signs[k] = -signs[k]
                
                # Update sums for all APs containing k
                idxs = pos_to_aps_arr[k]
                sums[idxs] += delta
                
                # Update G
                G += delta * C[:, k]
                
                steps += 1
                
                # Check hard max periodically
                if steps % 30 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_local_hard_max:
                        best_local_hard_max = current_hard_max
                        best_local_signs = signs.copy()
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
            # Annealing
            if steps % 500 == 0:
                T *= 0.97
                if T < T_min:
                    T = T_min
                    
            # Early stop if we hit 1
            if best_hard_max <= 1:
                break
=======
            # Delta for signs
            delta = -2.0 * signs[k]
            
            # Change in J = sum S_P^2
            delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
            
            # Combined objective: primarily sum-of-squares, with soft-max penalty
            # The soft-max term encourages reducing the worst AP sums
            idxs = pos_to_aps_arr[k]
            
            # Use histogram-based soft-max for faster evaluation
            if steps % 5 == 0:
                old_soft = soft_max_from_hist()
                # Compute new soft-max by temporarily updating histogram
                # For efficiency, approximate by checking if any AP with |sum| == current_max
                # would be reduced
                current_max = get_hard_max_from_hist()
                # Check if any of the affected APs have |sum| == current_max
                # and would be reduced by the flip
                max_reduced = False
                for idx in idxs:
                    old_val = int(sums[idx])
                    new_val = old_val + int(delta)
                    if abs(old_val) == current_max and abs(new_val) < current_max:
                        max_reduced = True
                        break
                # Approximate delta_soft: if max is reduced, soft-max decreases
                # Rough estimate: delta_soft ≈ -1.0 if max reduced, else 0
                if max_reduced:
                    delta_soft = -1.0
                else:
                    delta_soft = 0.0
                delta_combined = delta_J + 5.0 * delta_soft
            else:
                delta_combined = delta_J
            
            # Metropolis Criterion
            accept = False
            if delta_combined < 0:
                accept = True
            else:
                if delta_combined / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_combined / T)
                    
            if accept:
                # Apply move
                signs[k] = -signs[k]
                
                # Update sums and histogram for all APs containing k
                update_hist_flip(sums, idxs, delta)
                
                # Update G
                G += delta * C[:, k]
                
                steps += 1
                
                # Check hard max using histogram (O(1) amortized)
                if steps % 10 == 0:
                    current_hard_max = get_hard_max_from_hist()
                    if current_hard_max < best_local_hard_max:
                        best_local_hard_max = current_hard_max
                        best_local_signs = signs.copy()
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
            # Annealing
            if steps % 500 == 0:
                T *= 0.97
                if T < T_min:
                    T = T_min
                    
            # Early stop if we hit 1
            if best_hard_max <= 1:
                break
>>>>>>> REPLACE

