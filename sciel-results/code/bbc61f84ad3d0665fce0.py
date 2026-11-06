import math
import bisect
import time
import random
from fractions import Fraction

def construct():
    start = time.time()
    DEADLINE = start + 50.0  # Leave 10s margin for 60s limit

    # Calculate D = lcm(2..48)
    D = 1
    for d in range(2, 49):
        D = D * d // math.gcd(D, d)
        
    denoms = list(range(2, 49))
    w = [D // d for d in denoms]
    n = len(w)
    
    # Split into two halves
    n1 = n // 2  # 23
    n2 = n - n1  # 24
    
    # Weights for first half
    w1 = w[:n1]
    # Weights for second half
    w2 = w[n1:]
    
    # Generate packed values for second half
    # Packed value: (sum << 24) | mask
    MASK_BITS = 24
    MASK_LIMIT = (1 << MASK_BITS) - 1
    
    packed2 = [0]
    for i in range(n2):
        wi = w2[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) + mi
        # Optimization: Extend list in place or use comprehension?
        # Comprehension is faster in CPython for list ops
        packed2 = packed2 + [p + delta for p in packed2]
        
    if time.time() > DEADLINE:
        # Fallback: Heuristic
        return heuristic_search()

    # Sort second half
    packed2.sort()
    
    # Generate packed values for first half
    packed1 = [0]
    for i in range(n1):
        wi = w1[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) + mi
        packed1 = packed1 + [p + delta for p in packed1]
        
    if time.time() > DEADLINE:
        # If we ran out of time generating, we still have packed2 sorted.
        # We can do a quick search with a subset of packed1 or just fallback.
        # For safety, fallback to heuristic if we can't even sort packed1.
        return heuristic_search()

    # Sort first half
    packed1.sort()
    
    # Two-pointer search
    # i moves from 0 to len(packed1)-1
    # j moves from len(packed2)-1 to 0
    best_W = -1
    best_mask = 0
    
    len1 = len(packed1)
    len2 = len(packed2)
    
    i = 0
    j = len2 - 1
    
    while i < len1 and j >= 0:
        if time.time() > DEADLINE:
            break
            
        p1 = packed1[i]
        s1 = p1 >> MASK_BITS
        m1 = p1 & MASK_LIMIT
        
        p2 = packed2[j]
        s2 = p2 >> MASK_BITS
        m2 = p2 & MASK_LIMIT
        
        total_s = s1 + s2
        
        if total_s < D:
            if total_s > best_W:
                best_W = total_s
                best_mask = m1 | (m2 << n1)
            i += 1
        else:
            j -= 1
            
    if best_mask == 0:
        return heuristic_search()
        
    # Reconstruct indices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result

def heuristic_search():
    """
    Fallback local search if exact method times out or fails.
    Based on Inspiration 1.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    best_deficit = Fraction(1, 1)
    best_indices = []
    
    def get_deficit(indices):
        s = Fraction(0)
        for i in indices:
            s += Fraction(1, denominators[i])
        if s >= 1:
            return None
        return 1 - s

    def try_update(indices):
        nonlocal best_deficit, best_indices
        d = get_deficit(indices)
        if d is not None and d < best_deficit:
            best_deficit = d
            best_indices = list(indices)

    # Initial good heuristic
    # Greedy: sort by value descending, add if < 1
    # But we want to minimize deficit, so we want sum close to 1.
    # Greedy usually fills up close to 1.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        perm = list(range(n))
        random.shuffle(perm)
        
        current_indices = []
        current_sum = Fraction(0)
        
        for i in perm:
            val = Fraction(1, denominators[i])
            if current_sum + val < 1:
                current_indices.append(i)
                current_sum += val
        
        try_update(current_indices)
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current_indices)
            
            # Try adding
            for i in range(n):
                if i in current_set:
                    continue
                val = Fraction(1, denominators[i])
                if current_sum + val < 1:
                    new_sum = current_sum + val
                    if 1 - new_sum < best_deficit:
                        current_indices.append(i)
                        current_sum = new_sum
                        current_set.add(i)
                        try_update(current_indices)
                        improved = True
                        break
            
            # Try swapping
            if not improved:
                for i in range(n):
                    if i not in current_set:
                        continue
                    val_i = Fraction(1, denominators[i])
                    for j in range(n):
                        if j in current_set:
                            continue
                        val_j = Fraction(1, denominators[j])
                        new_sum = current_sum - val_i + val_j
                        if new_sum < 1:
                            if 1 - new_sum < best_deficit:
                                current_indices = [j if x == i else x for x in current_indices]
                                current_sum = new_sum
                                current_set.remove(i)
                                current_set.add(j)
                                try_update(current_indices)
                                improved = True
                                break
                    if improved:
                        break
                        
    return best_indices if best_indices else [0]
