import math
import time
import random

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 85  # 5-second safety margin
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] sharing a factor with 255
    candidates = [i for i in range(1, n+1) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    m = len(candidates)
    idx_255 = candidates.index(n)
    
    # Precompute compatibility using bitmasks
    # compat[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    compat = [0] * m
    for i in range(m):
        ci = candidates[i]
        for j in range(i+1, m):
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    def clique_size(bits):
        return bin(bits).count('1')
    
    def can_add(clique_bits, idx):
        """Check if vertex idx can join the clique"""
        return (clique_bits & ~compat[idx]) == 0
    
    # Baseline: all multiples of 3
    base_bits = 0
    for i in range(m):
        if candidates[i] % 3 == 0:
            base_bits |= (1 << i)
    best_bits = base_bits
    best_size = clique_size(best_bits)
    
    # Greedy clique building starting from 255
    def greedy_clique():
        bits = 1 << idx_255
        # Get all vertices compatible with current clique
        available = [i for i in range(m) if i != idx_255 and can_add(bits, i)]
        
        while available:
            # Pick vertex with max degree into current clique
            best_idx = -1
            best_deg = -1
            for idx in available:
                deg = clique_size(bits & compat[idx])
                if deg > best_deg:
                    best_deg = deg
                    best_idx = idx
            if best_idx == -1:
                break
            bits |= (1 << best_idx)
            # Update available
            available = [i for i in available if i != best_idx and can_add(bits, i)]
        
        return bits
    
    g_bits = greedy_clique()
    if clique_size(g_bits) > best_size:
        best_bits = g_bits
        best_size = clique_size(g_bits)
    
    # Local search: 1-swap
    def local_search(bits, max_iter=2000):
        size = clique_size(bits)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for rem in range(m):
                if rem == idx_255:
                    continue
                if not (bits & (1 << rem)):
                    continue
                temp = bits & ~(1 << rem)
                for add in range(m):
                    if add == rem or (temp & (1 << add)):
                        continue
                    if not can_add(temp, add):
                        continue
                    new_bits = temp | (1 << add)
                    new_size = clique_size(new_bits)
                    if new_size > size:
                        bits = new_bits
                        size = new_size
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return bits, size
    
    best_bits, best_size = local_search(best_bits)
    
    # Random restarts
    random.seed(42)
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Weighted random greedy
        bits = 1 << idx_255
        available = [i for i in range(m) if i != idx_255 and can_add(bits, i)]
        
        while available:
            # Weight by connections to current clique
            weights = []
            for idx in available:
                w = clique_size(bits & compat[idx]) + 1
                weights.append(w)
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen = available[0]
            for idx, w in zip(available, weights):
                cum += w
                if r <= cum:
                    chosen = idx
                    break
            bits |= (1 << chosen)
            available = [i for i in available if i != chosen and can_add(bits, i)]
        
        # Local improvement
        bits, size = local_search(bits, max_iter=500)
        
        if size > best_size:
            best_bits = bits
            best_size = size
    
    # Extract result
    result = [candidates[i] for i in range(m) if best_bits & (1 << i)]
    if n not in result:
        result.append(n)
    
    # Verify
    result_set = set(result)
    valid = True
    lst = list(result_set)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if math.gcd(lst[i], lst[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not result_set:
        return sorted(range(3, n+1, 3))
    
    return sorted(result_set)
