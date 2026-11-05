import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute sphere indices
    if radius == 1:
        sphere_size = 1 + q * (n - 1)
        sphere_indices = np.zeros((total_words, sphere_size), dtype=np.int32)
        
        # Precompute powers of q for index calculation
        powers = np.array([q ** i for i in range(n)], dtype=np.int64)
        
        for i in range(total_words):
            word = all_words[i]
            sphere_indices[i, 0] = i
            idx = 1
            for pos in range(n):
                orig = word[pos]
                base_val = orig * powers[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_val = val * powers[pos]
                    new_idx = i - base_val + new_val
                    sphere_indices[i, idx] = new_idx
                    idx += 1
    else:
        # General case: compute sphere by Hamming distance
        sphere_indices_list = []
        for i in range(total_words):
            word = all_words[i]
            sphere = [i]
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for values in itertools.product(range(1, q), repeat=dist):
                        new_word = word.copy()
                        for pos, offset in zip(positions, values):
                            new_word[pos] = (word[pos] + offset) % q
                        new_idx = 0
                        for j in range(n):
                            new_idx = new_idx * q + new_word[j]
                        if new_idx not in sphere:
                            sphere.append(new_idx)
            sphere_indices_list.append(sphere)
        max_sph = max(len(s) for s in sphere_indices_list)
        sphere_indices = np.zeros((total_words, max_sph), dtype=np.int32)
        for i, s in enumerate(sphere_indices_list):
            sphere_indices[i, :len(s)] = s

    def run_greedy_randomized(seed):
        random.seed(seed)
        is_uncovered = np.ones(total_words, dtype=bool)
        codewords = []
        
        while is_uncovered.any():
            # Compute coverage for all candidates
            coverage = is_uncovered[sphere_indices].sum(axis=1)
            
            # Find max coverage
            max_cov = coverage.max()
            if max_cov == 0:
                break
                
            # Get all indices with max coverage
            candidates = np.where(coverage == max_cov)[0]
            
            # Pick one randomly
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            # Add to codewords
            codewords.append(best_idx)
            
            # Update uncovered
            is_uncovered[sphere_indices[best_idx]] = False
            
        return codewords

    # Run multiple restarts
    best_codewords = None
    best_size = float('inf')
    
    start_time = time.time()
    
    # Run as many restarts as possible within time limit
    # 120s limit. Let's reserve 10s for overhead.
    for i in range(1000):
        if time.time() - start_time > 110:
            break
        cw = run_greedy_randomized(i)
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            
    # Convert indices to words
    if best_codewords is None:
        return []
        
    return [all_words[idx].tolist() for idx in best_codewords]
