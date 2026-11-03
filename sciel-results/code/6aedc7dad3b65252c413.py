    # Local Search Loop
    while _time_left() > 10.0:
        remove_redundant()
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
            
        # Try to find a valid swap
        # Pick up to 10 random codewords to try
        swapped = False
        for _ in range(10):
            if not selected:
                break
            c = rng.choice(list(selected))
            b_c = balls[c]
            critical_mask = (cov_count[b_c] == 1)
            critical_words = b_c[critical_mask]
            
            if len(critical_words) == 0:
                continue
                
            candidates = set(balls[critical_words[0]].tolist())
            for w in critical_words[1:]:
                candidates.intersection_update(balls[w].tolist())
            candidates -= selected
            
            if not candidates:
                continue
                
            c_prime = rng.choice(list(candidates))
            
            # Swap is valid by construction
            selected.discard(c)
            selected.add(c_prime)
            cov_count[balls[c]] -= 1
            cov_count[balls[c_prime]] += 1
            swapped = True
            break # Try removal after one swap
            
        if swapped:
            remove_redundant()
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
