<<<<<<< SEARCH
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy
        while uncovered:
            best_gain = -1
            best_c = -1
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break
                    
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]
            
        # Local Search
        code_set = local_search(set(code_indices), deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
            if best_size <= 186:
                break
=======
    for run in range(num_runs):
        if time.time() > deadline:
            break
        
        random.seed(run * 12345 + 67890)
        
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy
        while uncovered:
            best_gain = -1
            best_c = -1
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break
                    
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]
            
        # Local Search
        code_set = local_search(set(code_indices), deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
            if best_size <= 186:
                break
>>>>>>> REPLACE

