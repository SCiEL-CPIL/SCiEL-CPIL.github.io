import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute conflict masks
    conflict_mask = [0] * n
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            g = gcd(ai, aj)
            lcm_val = (ai // g) * aj
            if lcm_val <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    weights = [1.0 / candidates[i] for i in range(n)]
    
    def greedy(order):
        mask = 0
        for i in order:
            if not (mask & conflict_mask[i]):
                mask |= (1 << i)
        return mask
    
    def local_search(mask):
        cur_sum = 0.0
        for i in range(n):
            if mask & (1 << i):
                cur_sum += weights[i]
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            # Try 1-add
            for i in range(n):
                if not (mask & (1 << i)) and not (mask & conflict_mask[i]):
                    new_sum = cur_sum + weights[i]
                    if new_sum > cur_sum:
                        mask |= (1 << i)
                        cur_sum = new_sum
                        improved = True
                        break
            if time.time() > deadline - 5.0:
                break
            # Try 1-swap
            sel = [i for i in range(n) if mask & (1 << i)]
            unsel = [i for i in range(n) if not (mask & (1 << i))]
            for i in sel:
                if time.time() > deadline - 5.0:
                    break
                mw = mask & ~(1 << i)
                for j in unsel:
                    if not (mw & conflict_mask[j]):
                        new_sum = cur_sum - weights[i] + weights[j]
                        if new_sum > cur_sum:
                            mask = mw | (1 << j)
                            cur_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
        return mask, cur_sum
    
    best_mask = 0
    best_sum = 0.0
    
    # Greedy by weight descending
    order1 = sorted(range(n), key=lambda i: -weights[i])
    m = greedy(order1)
    m, s = local_search(m)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Multiple random restarts
    random.seed(42)
    while time.time() < deadline - 10.0:
        order = sorted(range(n), key=lambda i: -weights[i] + random.uniform(-0.01, 0.01))
        m = greedy(order)
        m, s = local_search(m)
        if s > best_sum:
            best_sum = s
            best_mask = m
    
    result = [candidates[i] for i in range(n) if best_mask & (1 << i)]
    if not result:
        return [256]
    return result
