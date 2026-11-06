import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 128
    
    def compute_counts(A):
        """Vectorized computation of pair sum counts."""
        indices = np.where(A)[0]
        if len(indices) == 0:
            return np.zeros(257, dtype=np.int64)
        # Create all unordered pairs (i, j) with i <= j
        i_idx, j_idx = np.meshgrid(indices, indices, indexing='ij')
        mask = i_idx <= j_idx
        valid_i = i_idx[mask]
        valid_j = j_idx[mask]
        sums = valid_i + valid_j
        counts = np.zeros(257, dtype=np.int64)
        np.add.at(counts, sums, 1)
        return counts
    
    def get_score(counts):
        """Compute score: peak * 1000 + invalid_count"""
        target_counts = counts[128:257]
        peak = int(np.max(target_counts))
        invalid_count = int(np.sum(target_counts == 0))
        return peak * 1000 + invalid_count, peak, invalid_count
    
    # Initial solution: full set
    A = np.ones(n + 1, dtype=bool)
    counts = compute_counts(A)
    current_score, current_peak, current_invalid = get_score(counts)
    
    best_A = A.copy()
    best_score = current_score
    best_peak = current_peak
    best_invalid = current_invalid
    
    # Try to find a good initial solution by removing elements
    # Start with full set and greedily remove elements that don't increase peak
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        # Try removing each element
        best_remove = -1
        best_remove_score = current_score
        for idx in range(n + 1):
            if not A[idx]:
                continue
            # Compute what counts would be if we remove idx
            # Subtract contributions of idx
            new_counts = counts.copy()
            for y in range(n + 1):
                if A[y] and y != idx:
                    new_counts[idx + y] -= 1
            new_counts[2 * idx] -= 1
            
            new_score, new_peak, new_invalid = get_score(new_counts)
            if new_score < best_remove_score:
                best_remove_score = new_score
                best_remove = idx
                best_new_counts = new_counts
                best_new_peak = new_peak
                best_new_invalid = new_invalid
        
        if best_remove == -1:
            break
        
        # Apply the best removal
        idx = best_remove
        for y in range(n + 1):
            if A[y] and y != idx:
                counts[idx + y] -= 1
        counts[2 * idx] -= 1
        A[idx] = False
        
        current_score = best_remove_score
        current_peak = best_new_peak
        current_invalid = best_new_invalid
        
        if current_score < best_score:
            best_score = current_score
            best_peak = current_peak
            best_invalid = current_invalid
            best_A = A.copy()
    
    # Simulated annealing with vectorized updates
    temperature = 5.0
    cooling = 0.9995
    min_temp = 0.01
    
    while time.time() < deadline - 2:
        if temperature < min_temp:
            # Try to restart with a slightly different initial solution
            if random.random() < 0.3:
                # Perturb best solution
                A = best_A.copy()
                # Add a few random elements
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, n)
                    A[idx] = True
                counts = compute_counts(A)
                current_score, current_peak, current_invalid = get_score(counts)
                temperature = 3.0
            else:
                break
        
        idx = random.randint(0, n)
        
        if A[idx]:
            # Remove idx
            for y in range(n + 1):
                if A[y] and y != idx:
                    counts[idx + y] -= 1
            counts[2 * idx] -= 1
            A[idx] = False
        else:
            # Add idx
            for y in range(n + 1):
                if A[y] and y != idx:
                    counts[idx + y] += 1
            counts[2 * idx] += 1
            A[idx] = True
            
        new_score, new_peak, new_invalid = get_score(counts)
        delta = new_score - current_score
        
        if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
            current_score = new_score
            current_peak = new_peak
            current_invalid = new_invalid
            
            if new_score < best_score:
                best_score = new_score
                best_peak = new_peak
                best_invalid = new_invalid
                best_A = A.copy()
        else:
            # Revert
            if A[idx]:
                # It was added, so remove it
                for y in range(n + 1):
                    if A[y] and y != idx:
                        counts[idx + y] -= 1
                counts[2 * idx] -= 1
                A[idx] = False
            else:
                # It was removed, so add it back
                for y in range(n + 1):
                    if A[y] and y != idx:
                        counts[idx + y] += 1
                counts[2 * idx] += 1
                A[idx] = True
        
        temperature *= cooling
    
    # Final polish: try to improve the best solution
    if best_invalid > 0:
        return list(range(n + 1))
    
    return [i for i in range(n + 1) if best_A[i]]
