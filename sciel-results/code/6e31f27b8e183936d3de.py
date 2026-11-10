import itertools
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    random.seed(42)
    
    n = P['n']
    m = P['m']
    
    # Generate all subsets as integer masks
    all_masks = []
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            all_masks.append(mask)
            
    # Helper to check comparability
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
        
    # Helper to count comparable sets in a list
    def count_comparable(target, family_list):
        count = 0
        for s in family_list:
            if is_comparable(target, s):
                count += 1
        return count

    # 1. Greedy Initialization on B12
    family = set()
    family.add(0) # Empty set
    
    # Add all singletons
    for v in range(n):
        family.add(1 << v)
        
    candidates = [mask for mask in all_masks if mask not in family]
    
    # We need m - len(family) more sets
    while len(family) < m:
        best_candidate = None
        best_gain = -1
        
        # Optimization: We can sample or iterate all. 4096 is small enough to iterate all.
        for c in candidates:
            gain = count_comparable(c, family)
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                
        if best_candidate is None:
            break
            
        family.add(best_candidate)
        candidates.remove(best_candidate)
        
    # Convert to list for SA
    family_list = list(family)
    
    # Calculate current score
    def calc_score(fam):
        score = 0
        k = len(fam)
        for i in range(k):
            for j in range(i+1, k):
                if is_comparable(fam[i], fam[j]):
                    score += 1
        return score
        
    current_score = calc_score(family_list)
    best_family = family_list[:]
    best_score = current_score
    
    # 2. Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    iterations = 20000
    
    for i in range(iterations):
        # Pick a random set in family to remove
        idx_in = random.randint(0, m - 1)
        u = family_list[idx_in]
        
        # Pick a random set not in family to add
        # To ensure v is not in family, we can pick from all_masks and retry if in family
        # Or maintain a list of outside sets. 
        # Simpler: pick random mask, check if in family. 
        # Since family is small (40) and universe large (4096), collision is rare.
        v = random.choice(all_masks)
        if v in family:
            # Try a few times
            for _ in range(10):
                v = random.choice(all_masks)
                if v not in family:
                    break
            else:
                continue # Skip if couldn't find a valid v
                
        # Calculate delta
        # Remove u, add v
        # New family is family_list with u replaced by v
        
        # Count comparable in (family \ {u}) for u and v
        # We can just iterate family_list and skip u
        
        gain_u = 0
        gain_v = 0
        for s in family_list:
            if s == u:
                continue
            if is_comparable(s, u):
                gain_u += 1
            if is_comparable(s, v):
                gain_v += 1
                
        delta = gain_v - gain_u
        
        if delta > 0 or random.random() < math.exp(delta / T):
            family_list[idx_in] = v
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = family_list[:]
                
        T *= cooling_rate
        
    # Convert best_family masks to lists of ints
    result = []
    for mask in best_family:
        s = []
        for v in range(n):
            if mask & (1 << v):
                s.append(v)
        result.append(s)
        
    return result
