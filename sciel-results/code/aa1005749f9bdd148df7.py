import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)
    best_family = []
    best_size = 0

    def greedy_build(order, current_best_size=0):
        """
        Greedily build a family from the given order of set indices.
        Uses an efficient data structure to detect sunflowers:
        For each set i in F, inter_maps[i] maps intersection_mask -> list of indices j
        such that F[i] & F[j] == intersection_mask.
        
        A new set m conflicts with existing F[i], F[j] if there exists i, j in F
        such that m & F[i] == m & F[j] == F[i] & F[j].
        """
        F = []
        inter_maps = []
        
        for pos, idx in enumerate(order):
            if time.time() > deadline:
                break
            
            m = all_sets[idx]
            conflict = False
            
            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflict = True
                            break
                if conflict:
                    break
            
            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})
                
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)
                
                if len(F) >= max_target:
                    break
                # Early termination heuristic: if we're far behind best and running out of time/sets
                # This is a rough check.
                if len(F) + (len(order) - pos) <= current_best_size:
                    break

        return F

    # 1. Deterministic pass (natural order)
    if time.time() < deadline:
        order = list(range(total))
        F = greedy_build(order, best_size)
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
    
    # 2. Random restarts with different seeds/orders
    rng = random.Random(12345)
    
    # Pre-generate a few different random orders to reuse? 
    # Or just shuffle on the fly. Shuffling 2184 items is fast.
    
    while time.time() < deadline - 2.0:  # Leave 2s for cleanup/return
        order = list(range(total))
        rng.shuffle(order)
        
        # Optional: Weighted shuffle or priority based on something?
        # For now, simple random restarts.
        
        F = greedy_build(order, best_size)
        
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
            if best_size >= max_target:
                break

    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
