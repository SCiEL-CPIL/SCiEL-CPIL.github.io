import time
import sys
from math import gcd
from functools import lru_cache

# Precompute prime factors and masks
# Primes up to 511
def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(511)
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES)}

# Precompute factor masks for 1..511
# mask[a] is an integer where bit i is 1 if PRIMES[i] divides a
FACTOR_MASKS = [0] * 512
for a in range(1, 512):
    m = a
    mask = 0
    for p in PRIMES:
        if p * p > m:
            break
        if m % p == 0:
            mask |= (1 << PRIME_INDEX[p])
            while m % p == 0:
                m //= p
    if m > 1:
        # m is a prime factor
        if m in PRIME_INDEX:
            mask |= (1 << PRIME_INDEX[m])
        else:
            # This shouldn't happen for m <= 511 if we have all primes up to 511
            # But just in case, we might need to handle it. 
            # Actually, if m > 1, it is prime. And since m <= 511, it must be in PRIMES.
            pass
    FACTOR_MASKS[a] = mask

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Vertices are 1..511
    vertices = list(range(1, N))
    weights = [0.0] * N
    for a in vertices:
        weights[a] = 1.0 / (N - a)
    
    # Sort vertices by weight descending (largest a first)
    sorted_vertices = sorted(vertices, key=lambda x: -weights[x])
    
    # 1. Greedy Construction
    selected = []
    selected_set = set()
    used_mask = 0
    
    for a in sorted_vertices:
        if (used_mask & FACTOR_MASKS[a]) == 0:
            selected.append(a)
            selected_set.add(a)
            used_mask |= FACTOR_MASKS[a]
            
    current_sum = sum(weights[a] for a in selected)
    best_sum = current_sum
    best_selected = list(selected)
    
    # 2. Local Search (Add and Swap)
    available = [a for a in vertices if a not in selected_set]
    
    # Helper to check if a is compatible with current used_mask
    # def is_compatible(a, used_mask): return (used_mask & FACTOR_MASKS[a]) == 0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        # We want to add a number that is compatible with used_mask.
        # To be efficient, we can just iterate through available and check.
        # Since available is ~450 items, and check is O(1), this is fast.
        best_add = -1
        best_add_weight = 0.0
        
        for a in available:
            if (used_mask & FACTOR_MASKS[a]) == 0:
                if weights[a] > best_add_weight:
                    best_add_weight = weights[a]
                    best_add = a
                    
        if best_add != -1:
            selected.append(best_add)
            selected_set.add(best_add)
            available.remove(best_add)
            used_mask |= FACTOR_MASKS[best_add]
            current_sum += weights[best_add]
            improved = True
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
            continue # Restart loop to check more adds/swaps
        
        # Try to swap
        # For each a in selected, try to replace with b in available
        # Condition: b is compatible with (used_mask ^ FACTOR_MASKS[a])
        # Gain: weights[b] - weights[a]
        best_swap_a = -1
        best_swap_b = -1
        best_gain = 0.0
        
        # Optimization: If the max weight in available is <= weights[a], skip?
        # We want max weights[b] - weights[a].
        # Let's precompute max available weight? Or just check inside.
        
        for a in selected:
            # mask without a
            mask_without_a = used_mask & ~FACTOR_MASKS[a]
            
            # Quick check: if max(available) weight <= weights[a], we can't gain?
            # Not strictly, because we might swap a high weight for a slightly lower weight 
            # if it allows adding another high weight later? No, swap is a single move.
            # If weights[b] <= weights[a], gain is <= 0. We only accept positive gain.
            
            # Iterate available
            for b in available:
                if weights[b] <= weights[a]:
                    continue
                # Check if b is compatible with mask_without_a
                if (mask_without_a & FACTOR_MASKS[b]) == 0:
                    gain = weights[b] - weights[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap_a = a
                        best_swap_b = b
                        
        if best_swap_a != -1:
            # Perform swap
            selected.remove(best_swap_a)
            selected_set.remove(best_swap_a)
            available.append(best_swap_a)
            
            selected.append(best_swap_b)
            selected_set.add(best_swap_b)
            available.remove(best_swap_b)
            
            used_mask = used_mask & ~FACTOR_MASKS[best_swap_a]
            used_mask |= FACTOR_MASKS[best_swap_b]
            
            current_sum += best_gain
            improved = True
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
            continue # Restart loop
        
        # No add, no swap.
        # Try perturbation: Remove a random/low weight element, then add best available.
        if time.time() < deadline - 5.0:
            if len(selected) > 1:
                # Pick the smallest weight element in selected
                min_idx = 0
                min_w = weights[selected[0]]
                for i in range(1, len(selected)):
                    if weights[selected[i]] < min_w:
                        min_w = weights[selected[i]]
                        min_idx = i
                        
                a_to_remove = selected[min_idx]
                
                # Find best b in available compatible with (used_mask ^ mask[a_to_remove])
                mask_without_remove = used_mask & ~FACTOR_MASKS[a_to_remove]
                best_b = -1
                best_w_b = 0.0
                for b in available:
                    if (mask_without_remove & FACTOR_MASKS[b]) == 0:
                        if weights[b] > best_w_b:
                            best_w_b = weights[b]
                            best_b = b
                            
                if best_b != -1 and weights[best_b] > weights[a_to_remove]:
                    selected.remove(a_to_remove)
                    selected_set.remove(a_to_remove)
                    available.append(a_to_remove)
                    
                    selected.append(best_b)
                    selected_set.add(best_b)
                    available.remove(best_b)
                    
                    used_mask = used_mask & ~FACTOR_MASKS[a_to_remove]
                    used_mask |= FACTOR_MASKS[best_b]
                    
                    current_sum = current_sum - weights[a_to_remove] + weights[best_b]
                    improved = True
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_selected = list(selected)
                    continue
            
            # If perturbation didn't work or wasn't applied, we are likely at a local optimum.
            # Break to save time.
            break
             
    # Final check: ensure we return the best found
    return best_selected
