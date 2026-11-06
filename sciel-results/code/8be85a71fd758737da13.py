import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
        
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums(weights):
        """Generate all distinct possible sums using numpy."""
        sums = np.array([0], dtype=np.int64)
        for wi in weights:
            sums = np.concatenate([sums - wi, sums, sums + wi])
        sums = np.unique(sums)
        return np.sort(sums)
    
    def backtrack_coeffs(weights, target_sum):
        """Backtrack to find coefficients that achieve target_sum."""
        coeffs = [0] * len(weights)
        # Build history of achievable sums as sets for O(1) lookup
        history = [set([0])]
        curr = {0}
        for wi in weights:
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            history.append(curr)
        
        current_sum = target_sum
        for i in range(len(weights) - 1, -1, -1):
            wi = weights[i]
            if current_sum - wi in history[i]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in history[i]:
                coeffs[i] = 0
            elif current_sum + wi in history[i]:
                coeffs[i] = -1
                current_sum += wi
            else:
                coeffs[i] = 0
        return coeffs

    sums1 = generate_sums(w1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    sums2 = generate_sums(w2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Two-pointer to find pair closest to 0 (excluding 0)
    i = 0
    j = len(sums2) - 1
    best_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    while i < len(sums1) and j >= 0:
        if time.time() > deadline - 5:
            break
        total = sums1[i] + sums2[j]
        if total != 0 and abs(total) < best_val:
            best_val = abs(total)
            best_s1 = int(sums1[i])
            best_s2 = int(sums2[j])
        if total < 0:
            i += 1
        else:
            j -= 1
                        
    if time.time() > deadline or best_s1 is None:
        return [0] * 28 + [1, -1]
        
    coeffs1 = backtrack_coeffs(w1, best_s1)
    coeffs2 = backtrack_coeffs(w2, best_s2)
    
    return coeffs1 + coeffs2
