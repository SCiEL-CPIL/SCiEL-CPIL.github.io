        idx = random.randint(0, num_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        # Apply change
        if old_r <= N:
            cover_count[old_r::p] -= 1
        if new_r <= N:
            cover_count[new_r::p] += 1
        residues[idx] = new_r
        
        new_size = np.sum(cover_count == 0)
        delta = new_size - current_size
        
        if delta > 0 or random.random() < math.exp(-delta / T):
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_residues = residues.copy()
        else:
            # Revert
            if new_r <= N:
                cover_count[new_r::p] -= 1
            if old_r <= N:
                cover_count[old_r::p] += 1
            residues[idx] = old_r
            
        T *= cooling_rate
