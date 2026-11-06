import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit
    
    n = 192
    universe = set(range(1, n + 1))
    
    # Helper to check validity of a set S
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            
            pair_map = {}
            for i in range(len(others)):
                xi = others[i]
                for j in range(i + 1, len(others)):
                    xj = others[j]
                    s = (xi + xj) % a
                    if s not in pair_map:
                        pair_map[s] = []
                    pair_map[s].append((i, j))
            
            for z_idx, z in enumerate(others):
                target = (-z) % a
                if target in pair_map:
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True

    # Initial greedy set (largest first)
    selected = []
    for x in range(n, 0, -1):
        candidate = set(selected)
        candidate.add(x)
        if is_valid(candidate):
            selected.append(x)
    
    S = set(selected)
    best_size = len(S)
    
    # Phase 1: Deterministic add-first
    U_not_S = list(universe - S)
    U_not_S.sort(reverse=True)
    for elem in U_not_S:
        if time.time() - start_time > time_limit:
            break
        S.add(elem)
        if is_valid(S):
            if len(S) > best_size:
                best_size = len(S)
        else:
            S.discard(elem)
    
    # Phase 2: Systematic local search - try all swaps that might help
    # First, try to find any add that works
    while time.time() - start_time < time_limit:
        U_not_S = list(universe - S)
        if not U_not_S:
            break
        
        found_add = False
        for elem in U_not_S:
            if time.time() - start_time > time_limit:
                break
            S.add(elem)
            if is_valid(S):
                found_add = True
                break
            else:
                S.discard(elem)
        
        if not found_add:
            # Try swaps
            S_list = list(S)
            found_swap = False
            for rem in S_list:
                if time.time() - start_time > time_limit:
                    break
                for add in U_not_S:
                    if time.time() - start_time > time_limit:
                        break
                    new_S = (S - {rem}) | {add}
                    if is_valid(new_S):
                        S = new_S
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if not found_swap:
                # Stuck, try random exploration
                if random.random() < 0.3:
                    rem = random.choice(S_list)
                    add = random.choice(U_not_S)
                    new_S = (S - {rem}) | {add}
                    if is_valid(new_S):
                        S = new_S
                else:
                    break
    
    # Phase 3: Final polish
    if time.time() - start_time < time_limit:
        U_not_S = list(universe - S)
        U_not_S.sort(reverse=True)
        for elem in U_not_S:
            if time.time() - start_time > time_limit:
                break
            S.add(elem)
            if is_valid(S):
                pass
            else:
                S.discard(elem)
    
    return list(S)

