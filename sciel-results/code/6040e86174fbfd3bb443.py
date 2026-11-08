    def local_search(res, load):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Relocation Moves
            for i in range(n_cust):
                current_f = res[i]
                d_i = demands[i]
                
                space = capacities - load
                feasible_f = np.where(space >= d_i)[0]
                
                if len(feasible_f) == 0:
                    continue
                    
                feasible_f = feasible_f[feasible_f != current_f]
                
                if len(feasible_f) == 0:
                    continue
                
                # Ensure feasible_f is a list/array of ints
                # Calculate cost delta for each feasible f
                base_delta = assign_cost[i, feasible_f] - assign_cost[i, current_f]
                
                # Open cost adjustments
                empty_mask = load[feasible_f] == 0
                # Handle case where empty_mask might be all False or True
                if np.any(empty_mask):
                    base_delta = base_delta.copy() # Avoid modifying view? base_delta is new array from subtraction
                    base_delta[empty_mask] += open_cost[feasible_f[empty_mask]]
                
                current_open_penalty = open_cost[current_f] if load[current_f] == d_i else 0.0
                
                total_deltas = base_delta - current_open_penalty
                
                # Ensure total_deltas is 1-D
                if total_deltas.ndim == 0:
                    total_deltas = np.array([total_deltas.item()])
                    
                min_idx = np.argmin(total_deltas)
                min_delta = total_deltas[min_idx]
                
                if min_delta < -1e-9:
                    f_new = feasible_f[min_idx]
                    load[current_f] -= d_i
                    load[f_new] += d_i
                    res[i] = f_new
                    improved = True
                    break
                    
            if improved:
                continue
            
            # 2. Swap Moves
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_i = res[i]
                d_i = demands[i]
                
                for k in range(i+1, n_cust):
                    f_k = res[k]
                    if f_i == f_k:
                        continue
                    d_k = demands[k]
                    
                    new_load_i = load[f_i] - d_i + d_k
                    if new_load_i > capacities[f_i] + 1e-9:
                        continue
                        
                    new_load_k = load[f_k] - d_k + d_i
                    if new_load_k > capacities[f_k] + 1e-9:
                        continue
                    
                    assign_delta = (assign_cost[i, f_k] - assign_cost[i, f_i]) + \
                                   (assign_cost[k, f_i] - assign_cost[k, f_k])
                    
                    open_delta = 0.0
                    if new_load_i == 0:
                        open_delta -= open_cost[f_i]
                    if new_load_k == 0:
                        open_delta -= open_cost[f_k]
                        
                    total_delta = assign_delta + open_delta
                    
                    if total_delta < -1e-9:
                        load[f_i] = new_load_i
                        load[f_k] = new_load_k
                        res[i] = f_k
                        res[k] = f_i
                        improved = True
                        break
                if improved:
                    break
