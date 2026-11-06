import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    # Evaluator kills at 90s wall-clock; use a conservative safety margin.
    time_limit = 75.0
    
    n = 192
    universe = set(range(1, n + 1))
    
    # Optimized validity check using pair sum maps
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            
            pair_map = defaultdict(list)
            for i in range(len(others)):
                xi = others[i]
                for j in range(i + 1, len(others)):
                    xj = others[j]
                    s = (xi + xj) % a
                    pair_map[s].append((i, j))
            
            for z_idx, z in enumerate(others):
                target = (-z) % a
                if target in pair_map:
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True

    # Initial greedy set
    selected = []
    for x in range(n, 0, -1):
        candidate = set(selected)
        candidate.add(x)
        if is_valid(candidate):
            selected.append(x)
    
    S = set(selected)
    best_S = set(S)
    best_size = len(S)
    
    # Local Search with multiple strategies
    # Strategy:
    # 1) Try to add elements in descending order
    # 2) If stuck, try swaps to unlock future additions
    # 3) Use simulated annealing-like approach with temperature

    max_iter = 5000  # increased iteration cap
    temperature = 1.0
    cooling_rate = 0.9995
    
    for iter_num in range(max_iter):
        if time.time() - start_time >= time_limit:
            break

        U_not_S = universe - S
        if not U_not_S:
            break

        # Try to add one element (any that keeps validity)
        added = False
        for x in sorted(U_not_S, reverse=True):
            if time.time() - start_time >= time_limit:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                added = True
                break

        if added:
            continue

        # If we cannot add, try a single swap to possibly unlock future adds.
        if not S:
            break

        U_not_S_list = list(U_not_S)
        if not U_not_S_list:
            break

        # Try multiple swap strategies
        found_swap = False
        
        # First try deterministic swaps: remove smallest, add largest
        S_list = sorted(S, reverse=True)
        if len(S_list) > 0:
            remove_candidate = S_list[-1]  # smallest
            add_candidate = U_not_S_list[0]  # largest not in S
            if remove_candidate != add_candidate:
                new_S = (S - {remove_candidate}) | {add_candidate}
                if is_valid(new_S):
                    S = new_S
                    found_swap = True
        
        # If deterministic didn't work, try random swaps
        if not found_swap:
            attempts = 50
            for _ in range(attempts):
                if time.time() - start_time >= time_limit:
                    break
                remove_candidate = random.choice(list(S))
                add_candidate = random.choice(U_not_S_list)
                if remove_candidate == add_candidate:
                    continue
                new_S = (S - {remove_candidate}) | {add_candidate}
                if is_valid(new_S):
                    S = new_S
                    found_swap = True
                    break
        
        # If no swap worked, try a 2-swap (remove 2, add 1) to reduce size temporarily
        # This might help escape local optima
        if not found_swap and len(S) > 2:
            S_list = list(S)
            for _ in range(20):
                if time.time() - start_time >= time_limit:
                    break
                if len(S_list) < 2:
                    break
                remove1, remove2 = random.sample(S_list, 2)
                add_candidate = random.choice(U_not_S_list)
                new_S = (S - {remove1, remove2}) | {add_candidate}
                if is_valid(new_S):
                    S = new_S
                    found_swap = True
                    break
        
        # Update temperature
        temperature *= cooling_rate

    # Final polish: try to add more elements
    U_not_S = universe - S
    for x in sorted(U_not_S, reverse=True):
        if time.time() - start_time >= time_limit:
            break
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)

    # Return the best set found
    if len(best_S) > len(S):
        S = best_S

    return list(S)
