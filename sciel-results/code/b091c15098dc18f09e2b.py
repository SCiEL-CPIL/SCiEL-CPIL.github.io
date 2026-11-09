import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    # Helper to check if a set is valid (no subset sums to 200)
    # Returns None if invalid, else the set of reachable sums
    def get_reachable_sums(S):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 in new_sums:
                return None
            reachable |= new_sums
        return reachable

    def valid(S):
        return get_reachable_sums(S) is not None

    # Greedy construction with a given order
    def greedy(order):
        selected = set()
        reachable = {0}
        for x in order:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_sums:
                selected.add(x)
                reachable |= new_sums
        return selected

    # Local search: try to add excluded elements
    def local_search_add(S):
        S = set(S)
        reachable = get_reachable_sums(S)
        if reachable is None:
            return S
        changed = True
        while changed:
            changed = False
            for x in range(1, 101):
                if x in S:
                    continue
                new_sums = {s + x for s in reachable if s + x <= 200}
                if 200 not in new_sums:
                    S.add(x)
                    reachable |= new_sums
                    changed = True
        return S

    # Local search: try 1-for-2 swaps (remove one, add two)
    def local_search_swap(S, time_limit=None):
        S = set(S)
        reachable = get_reachable_sums(S)
        if reachable is None:
            return S
        
        improved = True
        while improved:
            improved = False
            if time_limit and time.time() > time_limit:
                return S
            for rem in list(S):
                if time_limit and time.time() > time_limit:
                    return S
                temp_set = S - {rem}
                temp_reach = get_reachable_sums(temp_set)
                if temp_reach is None:
                    continue
                
                added = []
                for add in range(1, 101):
                    if add in S:
                        continue
                    new_sums = {s + add for s in temp_reach if s + add <= 200}
                    if 200 not in new_sums:
                        temp_reach |= new_sums
                        added.append(add)
                    if len(added) >= 2:
                        break
                
                if len(added) >= 2:
                    S = temp_set
                    for a in added:
                        S.add(a)
                    reachable = get_reachable_sums(S)
                    improved = True
        
        return S

    # Local search: try 1-for-1 swaps (remove one, add one)
    def local_search_swap11(S, time_limit=None):
        S = set(S)
        improved = True
        while improved:
            improved = False
            if time_limit and time.time() > time_limit:
                return S
            for rem in list(S):
                if time_limit and time.time() > time_limit:
                    return S
                temp_set = S - {rem}
                temp_reach = get_reachable_sums(temp_set)
                if temp_reach is None:
                    continue
                for add in range(1, 101):
                    if add in S:
                        continue
                    new_sums = {s + add for s in temp_reach if s + add <= 200}
                    if 200 not in new_sums:
                        S = temp_set | {add}
                        improved = True
                        break
        return S

    # Local search: try 2-for-1 swaps (remove two, add one) - reduces size but may enable further adds
    def local_search_swap21(S, time_limit=None):
        S = set(S)
        improved = True
        while improved:
            improved = False
            if time_limit and time.time() > time_limit:
                return S
            S_list = list(S)
            for i in range(len(S_list)):
                if time_limit and time.time() > time_limit:
                    return S
                for j in range(i+1, len(S_list)):
                    if time_limit and time.time() > time_limit:
                        return S
                    rem1, rem2 = S_list[i], S_list[j]
                    temp_set = S - {rem1, rem2}
                    temp_reach = get_reachable_sums(temp_set)
                    if temp_reach is None:
                        continue
                    # Try to add one element that wasn't in S
                    for add in range(1, 101):
                        if add in S:
                            continue
                        new_sums = {s + add for s in temp_reach if s + add <= 200}
                        if 200 not in new_sums:
                            S = temp_set | {add}
                            improved = True
                            break
                    if improved:
                        break
        return S

    # Initial solution: {51, ..., 100}
    current_set = set(range(51, 101))
    
    # Verify initial set
    if not valid(current_set):
        # Fallback to greedy
        current_set = greedy(range(100, 0, -1))
    
    best_set = set(current_set)
    best_size = len(current_set)

    # Try multiple greedy orders
    orders = [
        list(range(100, 0, -1)),   # high to low
        list(range(1, 101)),       # low to high
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        cand = greedy(order)
        cand = local_search_add(cand)
        cand = local_search_swap(cand, time_limit=deadline - 5)
        cand = local_search_swap11(cand, time_limit=deadline - 5)
        if len(cand) > best_size:
            best_set = cand
            best_size = len(cand)

    # Try randomized orders
    rng = random.Random(42)
    extra = 0
    while time.time() < deadline - 10.0 and extra < 20:
        order = list(range(1, 101))
        rng.shuffle(order)
        cand = greedy(order)
        cand = local_search_add(cand)
        cand = local_search_swap(cand, time_limit=deadline - 5)
        cand = local_search_swap11(cand, time_limit=deadline - 5)
        if len(cand) > best_size:
            best_set = cand
            best_size = len(cand)
        extra += 1

    # Final local search on best_set
    if time.time() < deadline - 5.0:
        best_set = local_search_add(best_set)
        best_set = local_search_swap(best_set, time_limit=deadline - 2)
        best_set = local_search_swap11(best_set, time_limit=deadline - 2)
        best_size = len(best_set)

    # Safety check
    if not valid(best_set):
        best_set = set(range(67, 101))
        best_size = len(best_set)

    return list(best_set)
