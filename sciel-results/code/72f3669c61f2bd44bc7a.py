"""Optimized sunflower-free 4-uniform family on [16]."""
import time
import random
from itertools import combinations

# Precompute all 4-subsets as bitmasks
ALL_SETS = []
ALL_SETS_SET = set()
for combo in combinations(range(16), 4):
    mask = 0
    for i in combo:
        mask |= (1 << i)
    ALL_SETS.append(mask)
    ALL_SETS_SET.add(mask)

N_SETS = len(ALL_SETS)  # 1820

def is_sunflower(a, b, c):
    """Check if three distinct sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    best_family = []
    best_size = 0
    
    random.seed(42)
    
    # Precompute for each pair of sets whether they can coexist with any third
    # Actually, let's just do randomized greedy with restarts
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random shuffle of all sets
        order = list(range(N_SETS))
        random.shuffle(order)
        
        family = []
        family_set = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            mask = ALL_SETS[idx]
            
            # Check against all pairs in family
            ok = True
            for i in range(len(family)):
                if not ok:
                    break
                for j in range(i+1, len(family)):
                    if is_sunflower(family[i], family[j], mask):
                        ok = False
                        break
            
            if ok:
                family.append(mask)
                family_set.add(mask)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        # Try local improvement: try to add more sets
        if time.time() > deadline:
            break
        
        # Try adding sets not in family
        for idx in range(N_SETS):
            if time.time() > deadline:
                break
            if idx in family_set:
                continue
            mask = ALL_SETS[idx]
            ok = True
            for i in range(len(family)):
                if not ok:
                    break
                for j in range(i+1, len(family)):
                    if is_sunflower(family[i], family[j], mask):
                        ok = False
                        break
            if ok:
                family.append(mask)
                family_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
