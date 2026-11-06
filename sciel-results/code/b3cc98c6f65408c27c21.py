<<<<<<< SEARCH
    def local_search_fix(colors, K, max_steps):
        # colors is a list or array of length 120
        # K is the number of colors allowed (0..K-1)
        # Try to reduce violations to 0
        
        # Track violations per K4? Too expensive to update fully.
        # Just sample violations.
        
        current_colors = colors[:]
        
        # Initial violation count
        # For speed, we can just iterate until no violations found in a sample or max steps.
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Find a violation
            # Sample random K4s
            found_violation = False
            # Check all if violations are low? No, sampling is faster.
            # But if we miss a violation, we might think we're done.
            # Let's do a full check every N steps or at the end?
            # Full check is 1820 * 6 ops ~ 11k ops. ~1ms.
            # 110s / 1ms = 110k full checks.
            
            # Let's just do a full check at the start of each "phase" or periodically.
            # For the inner loop, sample 100 K4s.
            
            sample_size = min(100, num_k4s)
            sample_indices = np.random.choice(num_k4s, sample_size, replace=False)
            
            viol_idx = -1
            for idx in sample_indices:
                if not is_valid_k4(current_colors, k4_edge_indices[idx]):
                    viol_idx = idx
                    break
            
            if viol_idx == -1:
                # No violation in sample. Verify fully.
                if count_violations(current_colors) == 0:
                    return current_colors, True
                # If there is a violation, we missed it. Try again.
                continue
            
            # Change a random edge in the violated K4
            edge_idxs = k4_edge_indices[viol_idx]
            e_idx = random.choice(edge_idxs)
            current_colors[e_idx] = random.randint(0, K - 1)
            
        # Final check
        if count_violations(current_colors) == 0:
            return current_colors, True
        return current_colors, False
=======
    def local_search_fix(colors, K, max_steps):
        # colors is a list or array of length 120
        # K is the number of colors allowed (0..K-1)
        # Try to reduce violations to 0
        
        current_colors = colors[:]
        
        # Track violations per K4
        violation_flags = np.zeros(num_k4s, dtype=bool)
        for k_idx, edge_idxs in enumerate(k4_edge_indices):
            violation_flags[k_idx] = not is_valid_k4(current_colors, edge_idxs)
        
        num_violations = violation_flags.sum()
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if num_violations == 0:
                return current_colors, True
            
            # Find a violated K4
            violated_indices = np.where(violation_flags)[0]
            if len(violated_indices) == 0:
                return current_colors, True
            
            viol_idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
            edge_idxs = k4_edge_indices[viol_idx]
            
            # Choose an edge to change - prefer edges that appear in many violated K4s
            edge_scores = {}
            for e_idx in edge_idxs:
                score = 0
                for k in edge_to_k4s[e_idx]:
                    if violation_flags[k]:
                        score += 1
                edge_scores[e_idx] = score
            
            # Choose the edge with highest score
            max_score = max(edge_scores.values())
            candidates = [e for e, s in edge_scores.items() if s >= max_score - 1]
            e_idx = random.choice(candidates)
            
            # Try different new colors and pick the one that reduces violations the most
            old_color = current_colors[e_idx]
            best_new_color = old_color
            best_violation_change = 0
            
            # Sample some colors to try
            if K <= 20:
                colors_to_try = list(range(K))
            else:
                colors_to_try = [random.randint(0, K - 1) for _ in range(min(10, K))]
            
            for new_color in colors_to_try:
                if new_color == old_color:
                    continue
                
                # Simulate the change
                violation_change = 0
                for k in edge_to_k4s[e_idx]:
                    if violation_flags[k]:
                        edge_idxs_k = k4_edge_indices[k]
                        new_colors = [current_colors[e] if e != e_idx else new_color for e in edge_idxs_k]
                        if is_valid_k4(new_colors, edge_idxs_k):
                            violation_change -= 1
                    else:
                        edge_idxs_k = k4_edge_indices[k]
                        new_colors = [current_colors[e] if e != e_idx else new_color for e in edge_idxs_k]
                        if not is_valid_k4(new_colors, edge_idxs_k):
                            violation_change += 1
                
                if violation_change < best_violation_change:
                    best_violation_change = violation_change
                    best_new_color = new_color
            
            # Apply the best color
            if best_new_color != old_color:
                for k in edge_to_k4s[e_idx]:
                    edge_idxs_k = k4_edge_indices[k]
                    new_colors = [current_colors[e] if e != e_idx else best_new_color for e in edge_idxs_k]
                    new_valid = is_valid_k4(new_colors, edge_idxs_k)
                    if violation_flags[k] and new_valid:
                        num_violations -= 1
                    elif not violation_flags[k] and not new_valid:
                        num_violations += 1
                    violation_flags[k] = not new_valid
                
                current_colors[e_idx] = best_new_color
            
            # Random perturbation to escape local optimum
            if best_new_color == old_color and random.random() < 0.1:
                random_e = random.randint(0, num_edges - 1)
                new_color = random.randint(0, K - 1)
                old_c = current_colors[random_e]
                if new_color != old_c:
                    for k in edge_to_k4s[random_e]:
                        edge_idxs_k = k4_edge_indices[k]
                        new_colors = [current_colors[e] if e != random_e else new_color for e in edge_idxs_k]
                        new_valid = is_valid_k4(new_colors, edge_idxs_k)
                        if violation_flags[k] and new_valid:
                            num_violations -= 1
                        elif not violation_flags[k] and not new_valid:
                            num_violations += 1
                        violation_flags[k] = not new_valid
                    current_colors[random_e] = new_color
        
        if count_violations(current_colors) == 0:
            return current_colors, True
        return current_colors, False
>>>>>>> REPLACE

