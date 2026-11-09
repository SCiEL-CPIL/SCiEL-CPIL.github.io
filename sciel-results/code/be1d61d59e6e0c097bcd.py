"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict graph using bitmasks for speed
    # conflict_mask[i] is a bitmask where bit j is set if numbers[i] and numbers[j] conflict
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:  # lcm <= 256, conflict
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Reciprocal weights
    weights = [1.0 / numbers[i] for i in range(n)]
    
    def can_add(i, selected_mask):
        """Check if node i can be added to the set represented by selected_mask."""
        return (conflict_mask[i] & selected_mask) == 0
    
    def get_conflict_count(i, selected_mask):
        """Count how many selected nodes conflict with i."""
        return bin(conflict_mask[i] & selected_mask).count('1')
    
    def reciprocal_sum_from_mask(mask):
        """Compute reciprocal sum from a bitmask."""
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total
    
    # Greedy construction with lookahead
    def greedy_construct(order):
        selected_mask = 0
        for i in order:
            if can_add(i, selected_mask):
                selected_mask |= (1 << i)
        return selected_mask
    
    # Try multiple greedy orderings and keep the best
    best_mask = 0
    best_sum = 0.0
    
    # Ordering 1: by weight descending (small numbers first)
    order1 = list(range(n))
    order1.sort(key=lambda i: (-weights[i], bin(conflict_mask[i]).count('1')))
    mask1 = greedy_construct(order1)
    sum1 = reciprocal_sum_from_mask(mask1)
    if sum1 > best_sum:
        best_sum = sum1
        best_mask = mask1
    
    # Ordering 2: by weight / (1 + conflict_count) - prefer nodes with fewer conflicts
    order2 = list(range(n))
    order2.sort(key=lambda i: (-(weights[i] / (1 + bin(conflict_mask[i]).count('1'))), -weights[i]))
    mask2 = greedy_construct(order2)
    sum2 = reciprocal_sum_from_mask(mask2)
    if sum2 > best_sum:
        best_sum = sum2
        best_mask = mask2
    
    # Ordering 3: random with some bias toward low numbers
    if elapsed() < DEADLINE - 5:
        for _ in range(5):
            if elapsed() > DEADLINE:
                break
            order3 = list(range(n))
            # Shuffle but bias toward lower indices (smaller numbers have higher weight)
            for i in range(n - 1, 0, -1):
                # With probability proportional to weight, swap
                j = random.randint(0, i)
                if random.random() < weights[order3[i]] / (weights[order3[i]] + weights[order3[j]] + 1e-10):
                    order3[i], order3[j] = order3[j], order3[i]
            mask3 = greedy_construct(order3)
            sum3 = reciprocal_sum_from_mask(mask3)
            if sum3 > best_sum:
                best_sum = sum3
                best_mask = mask3
    
    # Local search with bitmask operations
    selected_mask = best_mask
    
    def try_additions(mask):
        """Try to add any node that doesn't conflict. Returns (new_mask, improved)."""
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)):
                if can_add(i, new_mask):
                    new_mask |= (1 << i)
        return new_mask, (new_mask != mask)
    
    def try_swaps(mask):
        """Try removing one node and adding one or more others. Returns (best_new_mask, best_sum)."""
        current_sum = reciprocal_sum_from_mask(mask)
        best_local_mask = mask
        best_local_sum = current_sum
        
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if elapsed() > DEADLINE:
                break
            
            # Remove node i
            mask_without_i = mask & ~(1 << i)
            
            # Try to greedily add back as many as possible
            new_mask = mask_without_i
            for j in range(n):
                if not (new_mask & (1 << j)) and j != i:
                    if can_add(j, new_mask):
                        new_mask |= (1 << j)
            
            new_sum = reciprocal_sum_from_mask(new_mask)
            if new_sum > best_local_sum + 1e-12:
                best_local_sum = new_sum
                best_local_mask = new_mask
        
        return best_local_mask, best_local_sum
    
    # Iterative local search
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        
        # Try additions
        selected_mask, added = try_additions(selected_mask)
        if added:
            improved = True
            new_sum = reciprocal_sum_from_mask(selected_mask)
            if new_sum > best_sum:
                best_sum = new_sum
                best_mask = selected_mask
        
        if elapsed() > DEADLINE:
            break
        
        # Try swaps
        new_mask, new_sum = try_swaps(selected_mask)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_mask = new_mask
            selected_mask = new_mask
            improved = True
    
    # Convert back to numbers
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()
    return result
# EVOLVE-BLOCK-END
