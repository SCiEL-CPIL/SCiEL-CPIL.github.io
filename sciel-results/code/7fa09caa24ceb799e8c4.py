import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    return _construct_impl(q, n, radius)

def _construct_impl(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_space = q ** n
    
    # Precompute all centers and their balls
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    def get_ball_indices(center_idx):
        center = all_centers[center_idx]
        indices = [center_idx]
        if radius >= 1:
            for pos in range(n):
                original_val = center[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_word = center.copy()
                    new_word[pos] = new_val
                    indices.append(center_to_idx[tuple(new_word)])
        return np.array(indices, dtype=np.int32)
    
    balls = [get_ball_indices(i) for i in range(total_space)]
    ball_indices_matrix = np.array(balls, dtype=np.int32)
    
    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        coverage_count = np.zeros(total_space, dtype=np.int32)
        chosen_centers = []
        
        while True:
            if time.time() > deadline:
                return chosen_centers
            
            # A word is uncovered if coverage_count == 0
            uncovered_mask = (coverage_count == 0)
            
            # For each candidate center, count how many currently uncovered words it would cover
            scores = uncovered_mask[ball_indices_matrix].sum(axis=1)
            
            if scores.max() == 0:
                break
            
            # Find all indices with max score
            max_score = scores.max()
            candidates = np.where(scores == max_score)[0]
            
            # Pick one randomly
            best_idx = candidates[rng.integers(len(candidates))]
            
            chosen_centers.append(best_idx)
            coverage_count[balls[best_idx]] += 1
            
        return chosen_centers

    def remove_redundant(chosen_centers):
        if not chosen_centers:
            return chosen_centers
            
        chosen_centers = list(chosen_centers)
        
        # Build coverage count
        coverage_count = np.zeros(total_space, dtype=np.int32)
        for c in chosen_centers:
            coverage_count[balls[c]] += 1
        
        redundant_removed = True
        while redundant_removed and time.time() < deadline - 5.0:
            redundant_removed = False
            
            # Find a center whose removal keeps all its ball covered
            for i, c in enumerate(chosen_centers):
                if coverage_count[balls[c]].min() > 1:
                    # Remove this center
                    coverage_count[balls[c]] -= 1
                    chosen_centers.pop(i)
                    redundant_removed = True
                    break  # Restart scan from beginning
                    
        return chosen_centers

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    seed = 0
    while time.time() < deadline - 10.0:
        if time.time() > deadline:
            break
            
        code = greedy_cover(seed)
        code = remove_redundant(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
    
    if best_code is None:
        code = greedy_cover(0)
        best_code = remove_redundant(code)
        
    return [all_centers[i] for i in best_code]
