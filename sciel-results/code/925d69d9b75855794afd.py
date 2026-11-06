import random
import time
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n = len(all_sets)  # 1820
    
    best_size = 0
    best_family = []
    
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead
    
    # Helper to get all blockers of c in chosen
    # Returns a set of masks in chosen that form a sunflower with c and another set in chosen
    def get_blockers(c, chosen):
        blockers = set()
        # Group chosen sets by intersection with c
        buckets = {}
        for a in chosen:
            inter = a & c
            if inter not in buckets:
                buckets[inter] = []
            buckets[inter].append(a)
        
        for y, group in buckets.items():
            if len(group) < 2:
                continue
            g_len = len(group)
            # Check all pairs in the group
            for i in range(g_len):
                ai = group[i]
                ai_out = ai & ~y
                for j in range(i + 1, g_len):
                    aj = group[j]
                    aj_out = aj & ~y
                    if (ai_out & aj_out) == 0:
                        # ai and aj form a sunflower with c
                        blockers.add(ai)
                        blockers.add(aj)
        return blockers

    def mask_to_list(mask):
        return [i + 1 for i in range(16) if mask & (1 << i)]
    
    def greedy_least_conflicts():
        chosen = set()
        available = list(range(n))
        random.shuffle(available)
        
        while available:
            added = False
            min_conf = float('inf')
            candidates = []
            
            for idx in available:
                c = all_sets[idx]
                if c in chosen:
                    continue
                bl = get_blockers(c, chosen)
                if len(bl) == 0:
                    chosen.add(c)
                    available.remove(idx)
                    added = True
                    break
                elif len(bl) < min_conf:
                    min_conf = len(bl)
                    candidates = [idx]
                elif len(bl) == min_conf:
                    candidates.append(idx)
            
            if not added:
                break
                
        return chosen

    def local_search_sa(chosen, start_T=1.0, end_T=0.01, steps=2000):
        T = start_T
        cooling = (end_T / start_T) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            idx1 = random.randint(0, n - 1)
            c1 = all_sets[idx1]
            
            if c1 not in chosen:
                bl = get_blockers(c1, chosen)
                if len(bl) == 0:
                    chosen.add(c1)
                elif len(bl) == 1:
                    blocker = next(iter(bl))
                    if random.random() < 0.5:
                        chosen.remove(blocker)
                        chosen.add(c1)
            else:
                idx2 = random.randint(0, n - 1)
                c2 = all_sets[idx2]
                if c2 in chosen:
                    continue
                
                bl = get_blockers(c2, chosen)
                if len(bl) == 0:
                    chosen.add(c2)
                else:
                    # Check if c1 is the only blocker
                    if len(bl) == 1 and next(iter(bl)) == c1:
                        if random.random() < 0.5:
                            chosen.remove(c1)
                            chosen.add(c2)
            
            T *= cooling
            
        return chosen

    # Main search loop
    while time.time() - start_time < time_limit:
        # 1. Greedy Construction
        chosen = greedy_least_conflicts()
        
        # 2. Local Search
        chosen = local_search_sa(chosen)
        
        # Update best
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
            
    # Convert best_family to list of lists
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
    
    return result
