"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def construct():
    n = 128
    target_low = 128
    target_high = 256
    dim = 129  # indices 0..128
    
    start_time = time.time()
    time_limit = 55  # seconds
    
    # Precompute for each sum s in [128, 256] the list of pairs (a, b) with a <= b, a+b=s, 0<=a,b<=128
    # This allows vectorized count computation
    sum_pairs = {}
    for s in range(target_low, target_high + 1):
        a_min = max(0, s - n)
        a_max = min(n, s // 2)
        pairs = [(a, s - a) for a in range(a_min, a_max + 1)]
        sum_pairs[s] = pairs
    
    # For fast count computation, precompute for each element i, which sums it participates in
    # and the partner element
    # Actually, let's use a different approach: represent x as binary array, compute all pair sums
    
    def compute_counts(x):
        """x is a binary array of length 129. Returns array of counts for sums 128..256."""
        # Use FFT-like approach or direct convolution
        # Direct: for each s, count pairs
        counts = np.zeros(target_high - target_low + 1)
        for idx, s in enumerate(range(target_low, target_high + 1)):
            a_min = max(0, s - n)
            a_max = min(n, s // 2)
            # Count pairs (a, s-a) where both a and s-a are in A
            # x[a] * x[s-a] for a in [a_min, a_max]
            # But we need a <= s-a, which is a <= s/2, already enforced by a_max = s//2
            # For a == s-a (s even, a=s/2), it's one pair
            # For a < s-a, it's one pair (unordered)
            # So just sum x[a] * x[s-a] for a in [a_min, a_max]
            # But wait: if a != s-a, the pair (a, s-a) is counted once
            # If a == s-a, it's counted once
            # So the formula is correct
            count = 0
            for a in range(a_min, a_max + 1):
                b = s - a
                if x[a] == 1 and x[b] == 1:
                    count += 1
            counts[idx] = count
        return counts
    
    def objective(x_continuous):
        """Objective for differential evolution: x_continuous in [0,1]^129"""
        # Threshold to binary
        x_bin = (x_continuous > 0.5).astype(np.int8)
        counts = compute_counts(x_bin)
        # Check coverage
        uncovered = np.sum(counts == 0)
        max_count = np.max(counts) if len(counts) > 0 else 0
        # Penalty for uncovered sums
        penalty = 1000 * uncovered
        return max_count + penalty
    
    # Run differential evolution
    bounds = [(0, 1)] * dim
    
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=20,
            tol=0.01,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        best_x = result.x
    except Exception:
        # Fallback: use a simple heuristic
        best_x = np.zeros(dim)
        best_x[64:] = 1  # A = {64, 65, ..., 128}
    
    # Convert to binary set
    best_A = set(i for i in range(dim) if best_x[i] > 0.5)
    
    # Refine with local search
    def get_max_count_and_uncovered(A_set):
        max_count = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            count = 0
            a_min = max(0, s - n)
            a_max = min(n, s // 2)
            for a in range(a_min, a_max + 1):
                if a in A_set and (s - a) in A_set:
                    count += 1
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        return max_count, uncovered
    
    current_max, current_uncovered = get_max_count_and_uncovered(best_A)
    
    # If there are uncovered sums, try to fix them
    if current_uncovered > 0:
        # Add elements to cover uncovered sums
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            uncovered_sums = []
            for s in range(target_low, target_high + 1):
                count = 0
                a_min = max(0, s - n)
                a_max = min(n, s // 2)
                for a in range(a_min, a_max + 1):
                    if a in best_A and (s - a) in best_A:
                        count += 1
                if count == 0:
                    uncovered_sums.append(s)
            if not uncovered_sums:
                break
            # Pick a random uncovered sum and try to add an element to cover it
            s = uncovered_sums[np.random.randint(len(uncovered_sums))]
            a_min = max(0, s - n)
            a_max = min(n, s // 2)
            # Try adding a random a in [a_min, a_max] such that s-a is also in A or we add both
            candidates = []
            for a in range(a_min, a_max + 1):
                b = s - a
                if a not in best_A and b not in best_A:
                    candidates.append((a, b))
                elif a not in best_A:
                    candidates.append((a,))
                elif b not in best_A:
                    candidates.append((b,))
            if candidates:
                choice = candidates[np.random.randint(len(candidates))]
                for elem in choice:
                    best_A.add(elem)
                current_max, current_uncovered = get_max_count_and_uncovered(best_A)
    
    # Now refine: try to remove elements or swap to reduce max count
    import random
    best_max = current_max
    best_A_copy = set(best_A)
    
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing a random element
        if len(best_A) > 1:
            elem = random.choice(list(best_A))
            new_A = best_A - {elem}
            new_max, new_uncovered = get_max_count_and_uncovered(new_A)
            if new_uncovered == 0 and new_max <= current_max:
                best_A = new_A
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A_copy = set(best_A)
        
        # Try swapping: remove one, add another
        if len(best_A) > 0:
            elem = random.choice(list(best_A))
            candidates = [i for i in range(dim) if i not in best_A]
            if candidates:
                add_elem = random.choice(candidates)
                new_A = best_A - {elem} | {add_elem}
                new_max, new_uncovered = get_max_count_and_uncovered(new_A)
                if new_uncovered == 0 and new_max <= current_max:
                    best_A = new_A
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A_copy = set(best_A)
    
    return list(best_A)
# EVOLVE-BLOCK-END
