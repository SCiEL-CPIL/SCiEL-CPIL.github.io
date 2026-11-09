# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(1, n + 1), k):
        mask = 0
        for x in combo:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    total = len(all_sets)
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute intersection sizes between all pairs
    # inter[i][j] = popcount(all_sets[i] & all_sets[j])
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        for j in range(i + 1, total):
            v = popcount[all_sets[i] & all_sets[j]]
            inter[i][j] = v
            inter[j][i] = v
    
    def try_greedy(order):
        chosen = []  # list of indices
        chosen_set = set()
        
        for idx in order:
            conflict = False
            
            # Group chosen by intersection size with idx
            groups = {0: [], 1: [], 2: [], 3: []}
            for ci in chosen:
                t = inter[idx][ci]
                groups[t].append(ci)
            
            for t in range(4):
                group = groups[t]
                if len(group) < 2:
                    continue
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        if inter[group[i]][group[j]] == t:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen.append(idx)
                chosen_set.add(idx)
        
        return chosen
    
    # Try natural order
    best = try_greedy(range(total))
    
    # Try random orderings
    rng = random.Random(42)
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(total))
        rng.shuffle(order)
        result = try_greedy(order)
        if len(result) > len(best):
            best = result
    
    # Convert to sorted lists
    result = []
    for idx in best:
        combo = []
        mask = all_sets[idx]
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                combo.append(x)
        result.append(combo)
    
    return result
# EVOLVE-BLOCK-END
