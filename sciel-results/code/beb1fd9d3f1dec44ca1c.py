import itertools
import random
import numpy as np
import time
import math

def construct(q, n, radius):
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=int)
    total_words = len(all_words)
    
    # Precompute coverage masks using vectorized NumPy operations
    # dist[i, j] is Hamming distance between word i and word j
    # all_words[:, None, :] has shape (N, 1, n)
    # all_words[None, :, :] has shape (1, N, n)
    # Comparison gives (N, N, n) boolean array
    # Sum over last axis gives distances
    dist = np.sum(all_words[:, None, :] != all_words[None, :, :], axis=2)
    masks = dist <= radius
    
    def greedy_construct(seed):
        np.random.seed(seed)
        counts = np.zeros(total_words, dtype=int)
        codewords = []
        
        while True:
            if np.all(counts > 0):
                break
            
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
                
            # Compute scores: number of uncovered words covered by each candidate
            scores = np.sum(masks & uncovered_mask, axis=1)
            
            max_score = np.max(scores)
            if max_score == 0:
                break
            
            candidates = np.where(scores == max_score)[0]
            chosen_idx = candidates[np.random.randint(len(candidates))]
            
            codewords.append(chosen_idx)
            counts[masks[chosen_idx]] += 1
            
        return codewords

    def local_optimize(codewords, seed):
        """
        Given a set of codewords (indices), remove all redundant ones.
        Returns the minimized list.
        """
        np.random.seed(seed)
        # Initialize counts
        counts = np.zeros(total_words, dtype=int)
        for idx in codewords:
            counts[masks[idx]] += 1
            
        current_list = list(codewords)
        improved = True
        while improved:
            improved = False
            # Shuffle to avoid bias in deletion order
            order = list(range(len(current_list)))
            np.random.shuffle(order)
            
            for pos in order:
                if pos >= len(current_list):
                    continue
                idx = current_list[pos]
                
                # Tentatively remove
                counts[masks[idx]] -= 1
                
                if np.all(counts > 0):
                    # Can remove
                    current_list.pop(pos)
                    improved = True
                else:
                    # Restore
                    counts[masks[idx]] += 1
            
        return current_list

    def simulated_annealing(init_codewords, start_time, timeout):
        """
        Refine the solution using Simulated Annealing.
        Neighborhood: Add a random word, then locally optimize (delete redundancies).
        """
        np.random.seed(42) # Fixed seed for reproducibility of SA path, or use time? 
        # Let's use a random seed for SA to vary runs, but we call this multiple times?
        # Actually, we will call SA once with a good init.
        
        current = list(init_codewords)
        best = list(current)
        
        # Calculate initial counts for efficiency? 
        # local_optimize recalculates. 
        # For SA, we need to evaluate neighbors.
        # Neighbor generation:
        # 1. Pick random word r
        # 2. new_set = current + [r]
        # 3. optimized_set = local_optimize(new_set, seed)
        # 4. delta = len(optimized_set) - len(current)
        
        T = 1.0
        cool = 0.98
        iter_count = 0
        
        while T > 0.05 and time.time() - start_time < timeout:
            iter_count += 1
            
            # Generate neighbor
            # Add a random word
            r_idx = np.random.randint(total_words)
            if r_idx in current:
                # If it's already there, just try to optimize current? 
                # Or pick a different one.
                # Picking a different one is better for perturbation.
                # But if current is large, probability of collision is low.
                pass 
                
            new_set = current + [r_idx]
            
            # Local optimization
            # To make this faster, we can pass the current counts?
            # But local_optimize is pure function of indices.
            # Optimization: 
            # If we just added one word, we can update counts incrementally?
            # But local_optimize deletes multiple.
            # Let's stick to the pure function for correctness, it's fast enough for N=729.
            
            optimized_set = local_optimize(new_set, iter_count)
            
            delta = len(optimized_set) - len(current)
            
            # Acceptance criterion
            if delta < 0:
                current = optimized_set
                if len(current) < len(best):
                    best = list(current)
            else:
                # Accept with probability exp(-delta / T)
                if delta == 0:
                    # Accept neutral moves to wander
                    if np.random.random() < 0.5:
                        current = optimized_set
                else:
                    prob = math.exp(-delta / T)
                    if np.random.random() < prob:
                        current = optimized_set
            
            T *= cool
            
        return best

    # Main execution
    start_time = time.time()
    timeout = 115 # Leave 5s margin
    
    # 1. Get a good initial solution via Greedy
    # Run greedy a few times and pick best
    best_init = None
    best_init_size = float('inf')
    
    for seed in range(10):
        cw_indices = greedy_construct(seed)
        # Local optimize the greedy result immediately
        cw_indices = local_optimize(cw_indices, seed)
        if len(cw_indices) < best_init_size:
            best_init_size = len(cw_indices)
            best_init = cw_indices
            
        if time.time() - start_time > timeout - 20:
            break
            
    if best_init is None:
        return [list(w) for w in all_words]

    # 2. Run Simulated Annealing to improve
    # We can run multiple SA instances if time permits
    final_best = best_init
    final_best_size = len(best_init)
    
    # Run SA
    sa_result = simulated_annealing(best_init, start_time, timeout - 5)
    if len(sa_result) < final_best_size:
        final_best = sa_result
        final_best_size = len(sa_result)
        
    # If we still have time, try another SA run from a different init?
    # Or just run SA longer? The SA function uses time internally.
    # The current SA loop runs until T < 0.05 or time limit.
    
    return [list(all_words[i]) for i in final_best]
