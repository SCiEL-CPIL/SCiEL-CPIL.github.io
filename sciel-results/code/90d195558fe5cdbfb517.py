"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    random.seed(42)
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 128
    TARGET_LO = 128
    TARGET_HI = 256
    
    def count_pairs(A):
        """Count representations of each target sum using unordered pairs with equal summands allowed."""
        counts = np.zeros(TARGET_HI - TARGET_LO + 1, dtype=np.int32)
        A_sorted = sorted(A)
        n = len(A_sorted)
        for i in range(n):
            for j in range(i, n):
                s = A_sorted[i] + A_sorted[j]
                if TARGET_LO <= s <= TARGET_HI:
                    counts[s - TARGET_LO] += 1
        return counts
    
    def max_count(counts):
        return int(np.max(counts))
    
    def is_valid(counts):
        return np.all(counts > 0)
    
    def score(counts):
        if not is_valid(counts):
            return float('inf')
        return max_count(counts)
    
    # Initial solution: the seed
    A = set(range(64, 129))
    counts = count_pairs(A)
    best_score = score(counts)
    best_A = set(A)
    current_score = best_score
    current_A = set(A)
    
    def evaluate_move(A_set, add_elem=None, remove_elem=None):
        """Evaluate a move efficiently by updating counts incrementally."""
        nonlocal counts
        # We'll recompute from scratch for simplicity but use numpy
        return count_pairs(A_set)
    
    # Local search: try adding/removing elements
    # For each candidate element, compute the delta in counts
    def compute_delta_add(A_set, elem):
        """Compute how counts change when we add elem to A."""
        delta = np.zeros(TARGET_HI - TARGET_LO + 1, dtype=np.int32)
        for a in A_set:
            s = elem + a
            if TARGET_LO <= s <= TARGET_HI:
                delta[s - TARGET_LO] += 1
        s = elem + elem
        if TARGET_LO <= s <= TARGET_HI:
            delta[s - TARGET_LO] += 1
        return delta
    
    def compute_delta_remove(A_set, elem):
        """Compute how counts change when we remove elem from A."""
        delta = np.zeros(TARGET_HI - TARGET_LO + 1, dtype=np.int32)
        for a in A_set:
            if a != elem:
                s = elem + a
                if TARGET_LO <= s <= TARGET_HI:
                    delta[s - TARGET_LO] += 1
        s = elem + elem
        if TARGET_LO <= s <= TARGET_HI:
            delta[s - TARGET_LO] += 1
        return delta
    
    # Simulated annealing
    current_temp = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < deadline and current_temp > min_temp:
        # Try a random move: add or remove an element
        if len(A) < N and random.random() < 0.5:
            # Try adding an element
            candidates = [x for x in range(N + 1) if x not in A]
            if not candidates:
                continue
            elem = random.choice(candidates)
            delta = compute_delta_add(A, elem)
            new_counts = counts + delta
            new_score = score(new_counts)
            
            if new_score <= current_score or random.random() < np.exp(-(new_score - current_score) / max(current_temp, 1e-10)):
                A.add(elem)
                counts = new_counts
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(A)
        else:
            # Try removing an element
            if len(A) <= 1:
                continue
            # Prefer removing elements that contribute to high-count sums
            elem = random.choice(list(A))
            delta = compute_delta_remove(A, elem)
            new_counts = counts - delta
            new_score = score(new_counts)
            
            if new_score <= current_score or random.random() < np.exp(-(new_score - current_score) / max(current_temp, 1e-10)):
                A.remove(elem)
                counts = new_counts
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(A)
        
        current_temp *= cooling
        
        # Periodically try to repair if invalid
        if current_score == float('inf') and time.time() < deadline:
            # Add random elements to fix coverage
            for _ in range(20):
                if time.time() > deadline:
                    break
                candidates = [x for x in range(N + 1) if x not in A]
                if not candidates:
                    break
                elem = random.choice(candidates)
                A.add(elem)
            counts = count_pairs(A)
            current_score = score(counts)
    
    # Final: use best found
    return sorted(best_A)
# EVOLVE-BLOCK-END
