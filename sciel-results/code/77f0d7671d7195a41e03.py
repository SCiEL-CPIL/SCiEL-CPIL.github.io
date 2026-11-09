    def local_search(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # Initialize bad_k4s
        bad_k4s = set()
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            c = [colors[e] for e in e_indices]
            if len(set(c)) < 5:
                bad_k4s.add(k4_idx)
                
        current_bad = len(bad_k4s)
        
        if current_bad == 0:
            return colors
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if not bad_k4s:
                return colors
            
            # Pick a random bad K4
            # Convert to list for random choice, or use random index
            # Using list(set) is O(N).
            # Better: keep a list of bad k4s?
            # Let's just pick a random k4_idx from 0..num_k4s-1 and check if it's bad?
            # No, that's inefficient if few are bad.
            
            # Let's convert to list once per iteration?
            # Or just pick random from the set.
            # In Python, random.choice(list(bad_k4s)) is O(N).
            # Given N is small (max 1820), it's fine.
            
            bad_list = list(bad_k4s)
            k4_idx = random.choice(bad_list)
            e_indices = k4_edge_indices[k4_idx]
            
            # Try to change one of the edges in this K4
            e_idx = random.choice(e_indices)
            old_color = colors[e_idx]
            
            best_new_color = old_color
            best_delta = 0
            
            for new_color in range(K):
                if new_color == old_color:
                    continue
                
                # Calculate delta for this specific edge change
                # We need to check all K4s containing e_idx
                delta = 0
                for k4_j in edge_to_k4s[e_idx]:
                    e_j_indices = k4_edge_indices[k4_j]
                    
                    # Check if k4_j is currently bad
                    # We can check if it's in bad_k4s
                    # But we need to know if it *will* be bad.
                    
                    # Get current colors
                    c_old = [colors[e] for e in e_j_indices]
                    old_valid = len(set(c_old)) >= 5
                    
                    # Get new colors
                    c_new = []
                    for e in e_j_indices:
                        if e == e_idx:
                            c_new.append(new_color)
                        else:
                            c_new.append(colors[e])
                    new_valid = len(set(c_new)) >= 5
                    
                    if old_valid and not new_valid:
                        delta += 1
                    elif not old_valid and new_valid:
                        delta -= 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color

            if best_new_color != old_color:
                # Apply change
                colors[e_idx] = best_new_color
                
                # Update bad_k4s set
                # Only K4s in edge_to_k4s[e_idx] are affected
                for k4_j in edge_to_k4s[e_idx]:
                    e_j_indices = k4_edge_indices[k4_j]
                    c = [colors[e] for e in e_j_indices]
                    is_bad = len(set(c)) < 5
                    
                    if is_bad:
                        bad_k4s.add(k4_j)
                    else:
                        if k4_j in bad_k4s:
                            bad_k4s.remove(k4_j)
                            
                if not bad_k4s:
                    return colors
        
        return None
