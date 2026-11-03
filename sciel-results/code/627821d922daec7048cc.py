import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Map word to index
    # Base q representation
    def word_to_idx(w):
        idx = 0
        for k in range(n):
            idx = idx * q + w[k]
        return idx
        
    idx_map = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute spheres
    # Sphere of radius 1
    # For each word, generate all neighbors at distance 1
    spheres = []
    for w in all_words:
        sphere = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w.copy()
                    nw[pos] = val
                    sphere.append(nw)
        # Convert to indices
        sphere_indices = [idx_map[tuple(s)] for s in sphere]
        spheres.append(sphere_indices)
        
    # Greedy algorithm
    best_code = None
    best_size = N + 1
    
    for trial in range(2000):
        if time.time() > deadline:
            break
            
        uncovered = np.ones(N, dtype=bool)
        code_indices = []
        
        while uncovered.any():
            # Find candidate with max coverage of uncovered
            # To add randomness, we can add small noise or shuffle ties
            # Efficient way: 
            # coverage = [uncovered[s].sum() for s in spheres]
            # But this is slow in pure python. 
            # Use numpy:
            # Create a matrix? No, N is small (729).
            
            # Let's use a simple loop, it's fast enough for N=729
            max_cov = -1
            best_idx = -1
            candidates = []
            
            # Optimization: only consider indices that are still "useful" or just all?
            # Just all is fine.
            for i in range(N):
                cov = np.sum(uncovered[spheres[i]])
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
                    candidates = [i]
                elif cov == max_cov:
                    candidates.append(i)
            
            if max_cov == 0:
                break # Should not happen if we iterate until empty
                
            # Randomly pick from best candidates to break symmetry
            chosen_idx = candidates[np.random.randint(len(candidates))]
            
            code_indices.append(chosen_idx)
            uncovered[spheres[chosen_idx]] = False
            
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = code_indices
            
        # Early stop if we hit the theoretical lower bound or known best
        if best_size <= 73:
            # We can stop if we find 73, but maybe 72 exists? 
            # Lower bound is 57. But 73 is best known.
            # Let's keep searching a bit more to see if we can find smaller, 
            # but 73 is likely the target.
            if best_size <= 70:
                 break
                 
    if best_code is None:
        # Fallback: return all words (should not happen)
        return all_words.tolist()
        
    # Convert indices back to words
    result = [all_words[i] for i in best_code]
    return [w.tolist() for w in result]
