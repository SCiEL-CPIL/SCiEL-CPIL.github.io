import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 55.0

    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    n_all = len(all_sets)
    
    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)

    # Precompute inter_lists
    # inter_lists[i][k] = list of j such that all_sets[j] & all_sets[i] == k
    inter_lists = [None] * n_all
    for i in range(n_all):
        C = all_sets[i]
        d = {}
        for j in range(n_all):
            k = all_sets[j] & C
            if k not in d:
                d[k] = []
            d[k].append(j)
        inter_lists[i] = d

    def can_add_fast(C_idx, S_list, S_set):
        C = all_sets[C_idx]
        for A_idx in S_list:
            A = all_sets[A_idx]
            K = A & C
            candidates = inter_lists[C_idx].get(K, [])
            for B_idx in candidates:
                if B_idx in S_set:
                    B = all_sets[B_idx]
                    if (A & B) == K:
                        return False
        return True

    def greedy_pass(order, deadline):
        family = []
        in_family = set()
        for C_idx in order:
            if time.time() > deadline:
                break
            if can_add_fast(C_idx, family, in_family):
                family.append(C_idx)
                in_family.add(C_idx)
        return family, in_family

    best_family = []
    best_size = 0
    
    # Multiple greedy passes
    # 1. Random
    for _ in range(20):
        if time.time() > DEADLINE - 5.0:
            break
        order = list(range(n_all))
        random.shuffle(order)
        fam, _ = greedy_pass(order, DEADLINE - 5.0)
        if len(fam) > best_size:
            best_family = fam
            best_size = len(fam)
            
    # 2. Local search on best
    if best_family and time.time() < DEADLINE - 2.0:
        S_list = list(best_family)
        S_set = set(S_list)
        improved = True
        while improved and time.time() < DEADLINE - 2.0:
            improved = False
            # Try to add any set
            for C_idx in range(n_all):
                if C_idx in S_set:
                    continue
                if can_add_fast(C_idx, S_list, S_set):
                    S_list.append(C_idx)
                    S_set.add(C_idx)
                    improved = True
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_family = list(S_list)
                    break # Restart scan from beginning? Or continue?
                    # Continuing is better to find more.
                    # But if we break, we do another full pass.
                    # Let's not break.
            # If no improvement, try swaps?
            # Skip swaps for now to keep it simple and fast.
            
    return [mask_to_list(all_sets[i]) for i in best_family]
