import itertools
import random

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute the spheres as bitmasks
    # sphere_masks[i] is an int where bit j is 1 if word j is within distance radius of word i
    sphere_masks = [0] * total_words
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Helper to convert tuple of digits to index
    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Precompute all words' digits
    all_digits = [index_to_digits(i) for i in range(total_words)]
    
    # Generate spheres
    for i in range(total_words):
        mask = 0
        base_digits = all_digits[i]
        
        for r in range(radius + 1):
            for pos_subset in itertools.combinations(range(n), r):
                if r == 0:
                    mask |= (1 << i)
                    continue
                
                values = range(q)
                for vals in itertools.product(values, repeat=r):
                    new_digits = base_digits[:]
                    for k, pos in enumerate(pos_subset):
                        new_digits[pos] = vals[k]
                    new_idx = digits_to_index(new_digits)
                    mask |= (1 << new_idx)
                    
        sphere_masks[i] = mask

    all_mask = (1 << total_words) - 1
    
    best_codewords = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different random seeds for tie-breaking
    # 100 runs should be fast enough (each run is ~0.1-0.5s)
    for seed in range(100):
        rng = random.Random(seed)
        covered_mask = 0
        codewords = []
        
        while covered_mask != all_mask:
            best_idx = -1
            best_count = -1
            candidates = []
            
            for i in range(total_words):
                # Calculate new coverage
                # Optimization: If sphere is already fully covered, skip?
                # new_covered = sphere_masks[i] & (~covered_mask & all_mask)
                # But we need to find max, so we must check all.
                new_covered = sphere_masks[i] & (~covered_mask & all_mask)
                count = new_covered.bit_count()
                
                if count > best_count:
                    best_count = count
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            
            if best_count == 0:
                break # Should not happen if code is valid
            
            # Randomly select from candidates with max coverage
            best_idx = rng.choice(candidates)
            
            codewords.append(best_idx)
            covered_mask |= sphere_masks[best_idx]
            
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    # Convert indices back to list of lists
    result = []
    for idx in best_codewords:
        result.append(index_to_digits(idx))
        
    return result
