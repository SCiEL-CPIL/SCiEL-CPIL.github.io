import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin for 90s wall-clock limit
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflicts as bitmasks for O(1) compatibility checks
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute reciprocal values as Fractions for exact arithmetic
    recip_frac = [Fraction(1, numbers[i]) for i in range(n)]
    recip_float = [float(r) for r in recip_frac]
    
    def elapsed():
        return time.time() - start_time
    
    def mask_to_sum_float(mask):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += recip_float[i]
        return s
    
    def greedy_order(order, selected_mask, current_sum):
        """Add elements in given order if compatible. Returns (mask, sum_float)."""
        for i in order:
            if not (conflict_mask[i] & selected_mask):
                selected_mask |= (1 << i)
                current_sum += recip_float[i]
        return selected_mask, current_sum
    
    def local_search(selected_mask, current_sum):
        """Local search with bitmask operations. Returns (mask, sum_float)."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            
            # Try single swap: remove i, add best j
            for i in range(n):
                if not (selected_mask & (1 << i)):
                    continue
                new_mask = selected_mask & ~(1 << i)
                new_sum = current_sum - recip_float[i]
                # Find best j to add
                best_j = -1
                best_gain = 0.0
                for j in range(n):
                    if (new_mask & (1 << j)) or (conflict_mask[j] & new_mask):
                        continue
                    gain = recip_float[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0 and new_sum + best_gain > current_sum + 1e-12:
                    selected_mask = new_mask | (1 << best_j)
                    current_sum = new_sum + best_gain
                    improved = True
            
            # Try multi-add: remove i, add all compatible
            if not improved and elapsed() < DEADLINE:
                for i in range(n):
                    if not (selected_mask & (1 << i)):
                        continue
                    new_mask = selected_mask & ~(1 << i)
                    new_sum = current_sum - recip_float[i]
                    # Add all compatible in ascending order (higher reciprocal first)
                    for j in range(n):
                        if (new_mask & (1 << j)) or (conflict_mask[j] & new_mask):
                            continue
                        new_mask |= (1 << j)
                        new_sum += recip_float[j]
                    if new_sum > current_sum + 1e-12:
                        selected_mask = new_mask
                        current_sum = new_sum
                        improved = True
            
            # Try 2-swap: remove i1, i2, add best pair j1, j2
            if not improved and elapsed() < DEADLINE:
                for i1 in range(n):
                    if not (selected_mask & (1 << i1)):
                        continue
                    for i2 in range(i1 + 1, n):
                        if not (selected_mask & (1 << i2)):
                            continue
                        new_mask = selected_mask & ~(1 << i1) & ~(1 << i2)
                        new_sum = current_sum - recip_float[i1] - recip_float[i2]
                        
                        # Find best pair j1, j2 to add
                        best_pair_sum = 0.0
                        best_j1 = -1
                        best_j2 = -1
                        candidates = []
                        for j in range(n):
                            if (new_mask & (1 << j)) or (conflict_mask[j] & new_mask):
                                continue
                            candidates.append(j)
                        
                        for idx1 in range(len(candidates)):
                            j1 = candidates[idx1]
                            for idx2 in range(idx1 + 1, len(candidates)):
                                j2 = candidates[idx2]
                                if conflict_mask[j1] & (1 << j2):
                                    continue
                                pair_sum = recip_float[j1] + recip_float[j2]
                                if pair_sum > best_pair_sum:
                                    best_pair_sum = pair_sum
                                    best_j1 = j1
                                    best_j2 = j2
                        
                        if best_j1 >= 0 and best_j2 >= 0:
                            candidate_mask = new_mask | (1 << best_j1) | (1 << best_j2)
                            candidate_sum = new_sum + best_pair_sum
                            if candidate_sum > current_sum + 1e-12:
                                selected_mask = candidate_mask
                                current_sum = candidate_sum
                                improved = True
        
        return selected_mask, current_sum
    
    # Multiple greedy orderings + local search
    best_mask = 0
    best_sum = 0.0
    
    # Ordering 1: ascending
    order1 = list(range(n))
    m, s = greedy_order(order1, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Ordering 2: descending
    order2 = list(range(n - 1, -1, -1))
    m, s = greedy_order(order2, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Ordering 3: by 1/a / degree
    order3 = sorted(range(n), key=lambda i: recip_float[i] / (bin(conflict_mask[i]).count('1') + 1), reverse=True)
    m, s = greedy_order(order3, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Ordering 4: by degree only (fewest conflicts first)
    order4 = sorted(range(n), key=lambda i: bin(conflict_mask[i]).count('1'))
    m, s = greedy_order(order4, 0, 0.0)
    m, s = local_search(m, s)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Randomized restarts
    random.seed(42)
    restart_count = 0
    for _ in range(300):
        if elapsed() >= DEADLINE - 3:
            break
        restart_count += 1
        # Random order with bias toward high reciprocals
        weights = [recip_float[i] ** 2 for i in range(n)]
        order = sorted(range(n), key=lambda i: random.random() / (weights[i] + 1e-10))
        m, s = greedy_order(order, 0, 0.0)
        m, s = local_search(m, s)
        if s > best_sum:
            best_sum = s
            best_mask = m
    
    # Final polish: try adding any single element that fits
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        for j in range(n):
            if (best_mask & (1 << j)) or (conflict_mask[j] & best_mask):
                continue
            best_mask |= (1 << j)
            best_sum += recip_float[j]
            improved = True
    
    # Convert mask to list
    result = [numbers[i] for i in range(n) if (best_mask & (1 << i))]
    return result
