import time
from itertools import combinations

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 85

    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i

    def is_valid_set(s):
        """Check if set s is valid (no reciprocal subsumption)."""
        s_list = sorted(s, reverse=True)
        n = len(s_list)
        for i in range(n):
            target = weights[s_list[i]]
            # Check if target can be formed by subset of other elements
            others = [weights[s_list[j]] for j in range(n) if j != i]
            # Subset sum check
            possible = {0}
            for w in others:
                if w > target:
                    continue
                new_vals = set()
                for p in possible:
                    nv = p + w
                    if nv <= target:
                        new_vals.add(nv)
                possible.update(new_vals)
                if target in possible:
                    return False
        return True

    def check_add(s, new_elem):
        """Check if adding new_elem to set s keeps it valid."""
        if new_elem in s:
            return False
        # Check new_elem's constraint
        w_new = weights[new_elem]
        others = [weights[x] for x in s]
        possible = {0}
        for w in others:
            if w > w_new:
                continue
            new_vals = set()
            for p in possible:
                nv = p + w
                if nv <= w_new:
                    new_vals.add(nv)
            possible.update(new_vals)
        if w_new in possible:
            return False
        # Check each existing element's constraint with new_elem added
        for x in s:
            w_x = weights[x]
            others_x = [weights[y] for y in s if y != x] + [w_new]
            possible_x = {0}
            for w in others_x:
                if w > w_x:
                    continue
                new_vals = set()
                for p in possible_x:
                    nv = p + w
                    if nv <= w_x:
                        new_vals.add(nv)
                possible_x.update(new_vals)
            if w_x in possible_x:
                return False
        return True

    # Greedy construction with multiple orderings
    best_set = set()
    best_size = 0

    orderings = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),       # ascending
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    ]

    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        current = set()
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if check_add(current, x):
                current.add(x)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)

    # Local improvement: 1-for-1 swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(best_set):
                for y in range(24, 0, -1):
                    if y in best_set:
                        continue
                    trial = best_set - {x} | {y}
                    if is_valid_set(trial):
                        best_set = trial
                        improved = True
                        break
                if improved:
                    break

    # Local improvement: add elements
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for y in range(24, 0, -1):
                if y in best_set:
                    continue
                if check_add(best_set, y):
                    best_set.add(y)
                    improved = True
                    break

    # Local improvement: 1-for-2 swaps (remove one, add two)
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(best_set):
                trial_base = best_set - {x}
                for y1 in range(24, 0, -1):
                    if y1 in best_set:
                        continue
                    trial1 = trial_base | {y1}
                    if not is_valid_set(trial1):
                        continue
                    for y2 in range(24, 0, -1):
                        if y2 in trial1 or y2 == y1:
                            continue
                        trial2 = trial1 | {y2}
                        if is_valid_set(trial2):
                            best_set = trial2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Local improvement: 2-for-3 swaps (remove two, add three)
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            elements_list = list(best_set)
            for i in range(len(elements_list)):
                for j in range(i + 1, len(elements_list)):
                    x1, x2 = elements_list[i], elements_list[j]
                    trial_base = best_set - {x1, x2}
                    for y1 in range(24, 0, -1):
                        if y1 in best_set:
                            continue
                        trial1 = trial_base | {y1}
                        if not is_valid_set(trial1):
                            continue
                        for y2 in range(24, 0, -1):
                            if y2 in trial1 or y2 == y1:
                                continue
                            trial2 = trial1 | {y2}
                            if not is_valid_set(trial2):
                                continue
                            for y3 in range(24, 0, -1):
                                if y3 in trial2 or y3 == y1 or y3 == y2:
                                    continue
                                trial3 = trial2 | {y3}
                                if is_valid_set(trial3):
                                    best_set = trial3
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    if not best_set:
        best_set = {24}

    return list(best_set)
