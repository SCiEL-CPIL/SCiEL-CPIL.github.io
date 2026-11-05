import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin
    
    n = 256
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Precompute all 3-AP triples (a, b, c) where a < b < c and a+c = 2b
    triples = []
    for b in range(1, n + 1):
        for d in range(1, min(b, n - b) + 1):
            a = b - d
            c = b + d
            if a >= 1 and c <= n:
                triples.append((a, b, c))
    
    num_triples = len(triples)
    triple_arr = np.array(triples, dtype=np.int32)
    
    # Precompute conflicts: for each x, which other elements are in a triple with x
    # conflicts[x] is a list of (other1, other2) pairs such that {x, other1, other2} is a 3-AP
    # Actually, let's use a different structure: for each x, a bitset of elements that conflict with x
    # But we need to know if BOTH other elements are present.
    # Let's stick with the list of pairs approach but optimized.
    
    # conflicts[x] = list of (i, j) such that {x, i, j} is a 3-AP
    conflicts = [[] for _ in range(n + 1)]
    for (a, b, c) in triples:
        # a is in conflict with (b, c)
        conflicts[a].append((b, c))
        # b is in conflict with (a, c)
        conflicts[b].append((a, c))
        # c is in conflict with (a, b))
        conflicts[c].append((a, b))
    
    def can_add(x, mask):
        for (i, j) in conflicts[x]:
            if (mask >> i) & 1 and (mask >> j) & 1:
                return False
        return True
    
    def reciprocal_sum(mask):
        s = 0.0
        for x in range(1, n + 1):
            if (mask >> x) & 1:
                s += recip[x]
        return s
    
    # Greedy initialization
    mask = 0
    for x in range(1, n + 1):
        if can_add(x, mask):
            mask |= (1 << x)
    
    best_mask = mask
    best_sum = reciprocal_sum(mask)
    
    # Local search: try swaps and additions
    def local_search(current_mask, current_sum):
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            
            # Try adding any x not in set
            for x in range(1, n + 1):
                if (current_mask >> x) & 1:
                    continue
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << x)
                    new_sum = current_sum + recip[x]
                    if new_sum > current_sum + 1e-12:
                        current_mask = new_mask
                        current_sum = new_sum
                        improved = True
            
            # Try swaps: remove y, add x
            if time.time() - start_time > DEADLINE:
                break
                
            for y in range(1, n + 1):
                if not ((current_mask >> y) & 1):
                    continue
                for x in range(1, n + 1):
                    if (current_mask >> x) & 1:
                        continue
                    temp_mask = current_mask & ~(1 << y)
                    if can_add(x, temp_mask):
                        new_mask = temp_mask | (1 << x)
                        new_sum = current_sum - recip[y] + recip[x]
                        if new_sum > current_sum + 1e-12:
                            current_mask = new_mask
                            current_sum = new_sum
                            improved = True
                            break
                if time.time() - start_time > DEADLINE:
                    break
        
        return current_mask, current_sum
    
    best_mask, best_sum = local_search(best_mask, best_sum)
    
    # Try to improve using a more sophisticated approach
    # We can try to find a better set by considering the structure of 3-AP-free sets
    # For example, the base-3 representation without digit 2 is a classic construction
    
    # Let's try a few different initializations and see which one gives a better result
    # Initialization 1: Greedy (already done)
    # Initialization 2: Base-3 construction
    mask2 = 0
    for x in range(1, n + 1):
        # Check if x in base 3 has no digit 2
        valid = True
        t = x
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            mask2 |= (1 << x)
    
    sum2 = reciprocal_sum(mask2)
    if sum2 > best_sum:
        best_mask, best_sum = local_search(mask2, sum2)
    
    # Initialization 3: Randomized greedy with different orderings
    import random
    random.seed(42)
    
    for _ in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        mask3 = 0
        for x in order:
            if can_add(x, mask3):
                mask3 |= (1 << x)
        
        sum3 = reciprocal_sum(mask3)
        if sum3 > best_sum:
            best_mask, best_sum = local_search(mask3, sum3)
    
    # Final local search to polish
    best_mask, best_sum = local_search(best_mask, best_sum)
    
    # Extract result
    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return result
