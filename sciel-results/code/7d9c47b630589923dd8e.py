def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    
    aps = []
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)
    
    # Pre-compute distinct counts helper
    def calc_distinct(ap_colors):
        # ap_colors: (M, 4)
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        d1 = (c0 != c1)
        d2 = (c0 != c2) & (c1 != c2)
        d3 = (c0 != c3) & (c1 != c3) & (c2 != c3)
        return 1 + d1 + d2 + d3

    # Initial setup
    ap_indices_for_pos = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_for_pos[pos].append(ap_idx)
            
    def sa_search(k, time_limit):
        colors = np.random.randint(0, k, N)
        
        # Initial violation state
        ap_colors = colors[aps_np]
        distincts = calc_distinct(ap_colors)
        violations = distincts < 3
        total_viol = int(np.sum(violations))
        
        best_colors = colors.copy()
        best_viol = total_viol
        
        if best_viol == 0:
            return best_colors
            
        T = 1.0
        cooling_rate = 0.99995
        min_T = 0.001
        steps_since_improve = 0
        max_steps_no_improve = 5000
        
        end_time = start_time + time_limit
        
        while time.time() < end_time:
            if best_viol == 0:
                return best_colors
                
            if steps_since_improve > max_steps_no_improve:
                T = 1.0
                steps_since_improve = 0
                
            idx = np.random.randint(N)
            old_c = colors[idx]
            new_c = np.random.randint(k)
            
            if new_c == old_c:
                continue
                
            # Get affected APs
            affected = ap_indices_for_pos[idx]
            if not affected:
                continue
                
            # Get current violation status of affected APs
            old_viol_mask = violations[affected]
            old_viol_count = int(np.sum(old_viol_mask))
            
            # Apply change
            colors[idx] = new_c
            
            # Recompute distincts for affected APs
            # We need the colors of the affected APs
            # affected is a list of indices into aps_np
            # aps_np[affected] gives the positions
            # colors[aps_np[affected]] gives the colors
            aff_aps = aps_np[affected]
            aff_colors = colors[aff_aps]
            new_distincts = calc_distinct(aff_colors)
            new_viol_mask = new_distincts < 3
            new_viol_count = int(np.sum(new_viol_mask))
            
            delta = new_viol_count - old_viol_count
            
            if delta <= 0 or (T > 0 and np.random.random() < math.exp(-delta / T)):
                # Accept
                total_viol += delta
                violations[affected] = new_viol_mask
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    steps_since_improve = 0
                    if best_viol == 0:
                        return best_colors
                else:
                    steps_since_improve += 1
            else:
                # Reject
                colors[idx] = old_c
                steps_since_improve += 1
                
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        if best_viol == 0:
            return best_colors
        return None

    # Try 4 colors
    solution = sa_search(4, 25.0)
    if solution is not None:
        return solution.tolist()
        
    # Try 5 colors
    solution = sa_search(5, 15.0)
    if solution is not None:
        return solution.tolist()
        
    # Try 6 colors
    solution = sa_search(6, 10.0)
    if solution is not None:
        return solution.tolist()
        
    # Fallback
    return [0]*N
