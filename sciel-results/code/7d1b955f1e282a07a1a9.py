import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    random.seed(42)
    
    all_bitmasks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_bitmasks.append(mask)
    
    n = len(all_bitmasks)
    
    def is_valid_with(family, s):
        if len(family) < 2:
            return True
        
        groups = {}
        for a in family:
            inter = a & s
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(a)
        
        for T, group in groups.items():
            glen = len(group)
            if glen < 2:
                continue
            tails = [a & ~s for a in group]
            for i in range(glen):
                ti = tails[i]
                for j in range(i + 1, glen):
                    if ti & tails[j] == 0:
                        return False
        return True
    
    best_family = []
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        shuffled = list(range(n))
        random.shuffle(shuffled)
        
        family = []
        
        for idx in shuffled:
            s = all_bitmasks[idx]
            if is_valid_with(family, s):
                family.append(s)
        
        if len(family) > len(best_family):
            best_family = family[:]
    
    result = []
    for mask in best_family:
        elems = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(elems)
    result.sort()
    return result
