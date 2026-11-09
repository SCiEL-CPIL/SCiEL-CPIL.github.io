"""Improved program using Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import math
import time

def construct():
    n = 12
    K = 40
    total_masks = 1 << n
    
    # Initialize with random distinct masks
    chosen = []
    chosen_set = set()
    while len(chosen) < K:
        m = random.getrandbits(n)
        if m not in chosen_set:
            chosen.append(m)
            chosen_set.add(m)
            
    # Calculate initial contribs and score
    contrib = [0] * K
    for i in range(K):
        for j in range(i + 1, K):
            m_i = chosen[i]
            m_j = chosen[j]
            # Strict subset check: (a & b) == a and a != b
            # Since distinct, (a & b) == a implies a is strict subset of b
            if (m_i & m_j) == m_i or (m_i & m_j) == m_j:
                contrib[i] += 1
                contrib[j] += 1
                
    score = sum(contrib) // 2
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.99995
    min_T = 0.01
    
    start_time = time.time()
    max_time = 55.0 # Leave buffer for evaluator overhead
    
    while True:
        if time.time() - start_time > max_time:
            break
        if T < min_T:
            break
            
        i = random.randint(0, K - 1)
        old_m = chosen[i]
        
        # Choose neighborhood: 80% Hamming distance 1, 20% Global
        if random.random() < 0.8:
            bit = random.randint(0, n - 1)
            new_m = old_m ^ (1 << bit)
        else:
            new_m = random.getrandbits(n)
            
        if new_m == old_m:
            continue
            
        if new_m in chosen_set:
            continue
            
        # Calculate delta
        new_contrib_i = 0
        delta = 0
        changes = []
        
        for j in range(K):
            if j == i:
                continue
            m_j = chosen[j]
            
            # Inline is_comparable
            was_comp = ((old_m & m_j) == old_m) or ((old_m & m_j) == m_j)
            is_comp = ((new_m & m_j) == new_m) or ((new_m & m_j) == new_m)
            
            if is_comp:
                new_contrib_i += 1
                
            d = (1 if is_comp else 0) - (1 if was_comp else 0)
            if d != 0:
                changes.append((j, d))
                delta += d
                
        # Acceptance
        accept = False
        if delta > 0:
            accept = True
        else:
            if T > 0:
                exponent = delta / T
                if exponent > -700:
                    p = math.exp(exponent)
                else:
                    p = 0.0
                if random.random() < p:
                    accept = True
                    
        if accept:
            chosen[i] = new_m
            chosen_set.remove(old_m)
            chosen_set.add(new_m)
            score += delta
            contrib[i] = new_contrib_i
            for j, d in changes:
                contrib[j] += d
                
        T *= cooling
        
    # Convert masks to lists of integers
    result = []
    for m in chosen:
        s = []
        for i in range(n):
            if (m >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
