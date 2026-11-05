def construct():
    import time
    import random

    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s limit

    N = 180

    # Precompute all forbidden triples: (a, b, c) with a < b < c and a*(b+c) == b*c
    # For each element x, store the set of (u,v) pairs such that {x,u,v} is a forbidden triple
    conflict_sets = {i: set() for i in range(1, N + 1)}

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            if diff == 0:
                continue
            num = a * b
            if num % diff != 0:
                continue
            c = num // diff
            if c > b and c <= N:
                conflict_sets[a].add((b, c))
                conflict_sets[b].add((a, c))
                conflict_sets[c].add((a, b))

    # Precompute conflict counts
    conflict_counts = {i: len(conflict_sets[i]) for i in range(1, N + 1)}

    def can_add(x, in_set):
        for (u, v) in conflict_sets[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    def greedy_add(in_set, result, order):
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                result.add(x)

    def try_add_all(in_set, result, candidates):
        changed = True
        while changed:
            changed = False
            for y in candidates:
                if in_set[y]:
                    continue
                if can_add(y, in_set):
                    in_set[y] = True
                    result.add(y)
                    changed = True

    def local_search_1_for_2(in_set, result, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            sorted_result = sorted(result, key=lambda x: conflict_counts[x], reverse=True)
            for x in sorted_result:
                if time.time() > deadline:
                    break
                in_set[x] = False
                result.discard(x)
                added = 0
                for y in range(1, N + 1):
                    if in_set[y]:
                        continue
                    if can_add(y, in_set):
                        in_set[y] = True
                        result.add(y)
                        added += 1
                if added > 1:
                    improved = True
                elif added == 0:
                    in_set[x] = True
                    result.add(x)
                else:
                    improved = True

    def local_search_1_for_1_swap(in_set, result, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            sorted_result = sorted(result, key=lambda x: conflict_counts[x], reverse=True)
            for x in sorted_result:
                if time.time() > deadline:
                    break
                in_set[x] = False
                result.discard(x)
                best_y = None
                best_conflicts = float('inf')
                for y in range(1, N + 1):
                    if in_set[y]:
                        continue
                    if can_add(y, in_set):
                        c_count = conflict_counts[y]
                        if c_count < best_conflicts:
                            best_conflicts = c_count
                            best_y = y
                if best_y is not None:
                    in_set[best_y] = True
                    result.add(best_y)
                    improved = True
                else:
                    in_set[x] = True
                    result.add(x)

    def shake_and_recover(in_set, result, k, deadline):
        if len(result) < k + 2:
            return False
        sorted_result = sorted(result, key=lambda x: conflict_counts[x], reverse=True)
        removed = random.sample(sorted_result, k)
        for x in removed:
            in_set[x] = False
            result.discard(x)
        candidates = [y for y in range(1, N + 1) if not in_set[y]]
        candidates.sort(key=lambda x: conflict_counts[x])
        greedy_add(in_set, result, candidates)
        try_add_all(in_set, result, range(1, N + 1))
        return len(result) >= len(set(removed))

    best_size = 0
    best_result = None
    best_in_set = None

    def update_best(in_set, result):
        nonlocal best_size, best_result, best_in_set
        if len(result) > best_size:
            best_size = len(result)
            best_result = set(result)
            best_in_set = in_set[:]

    # Strategy 1: Start with high range, add lower descending
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(90, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 2: Start with high range, add lower ascending
    in_set = [False] * (N + 1)
    result = set()
    for i in range(91, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(1, 91))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 3: Pure greedy from high to low
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(N, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 4: Pure greedy from low to high
    in_set = [False] * (N + 1)
    result = set()
    greedy_add(in_set, result, range(1, N + 1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Strategy 5: Start with top 60
    in_set = [False] * (N + 1)
    result = set()
    for i in range(121, N + 1):
        in_set[i] = True
        result.add(i)
    greedy_add(in_set, result, range(120, 0, -1))
    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Randomized restarts
    random.seed(42)
    num_restarts = 0
    max_restarts = 200

    while time.time() < DEADLINE - 15 and num_restarts < max_restarts:
        num_restarts += 1
        in_set = [False] * (N + 1)
        result = set()

        seed_size = random.randint(30, 80)
        weights = [1.0 / i for i in range(1, N + 1)]
        seeds = set()
        remaining = list(range(1, N + 1))
        for _ in range(seed_size):
            if not remaining:
                break
            w_rem = [weights[i-1] for i in remaining]
            total = sum(w_rem)
            r = random.random() * total
            cum = 0
            idx = 0
            for j, w in enumerate(w_rem):
                cum += w
                if r <= cum:
                    idx = j
                    break
            seeds.add(remaining.pop(idx))

        for i in seeds:
            in_set[i] = True
            result.add(i)

        candidates = [i for i in range(1, N + 1) if not in_set[i]]
        random.shuffle(candidates)
        greedy_add(in_set, result, candidates)
        try_add_all(in_set, result, range(1, N + 1))
        update_best(in_set, result)

        if time.time() < DEADLINE - 15:
            in_set2 = [False] * (N + 1)
            result2 = set()
            candidates2 = list(range(1, N + 1))
            random.shuffle(candidates2)
            greedy_add(in_set2, result2, candidates2)
            try_add_all(in_set2, result2, range(1, N + 1))
            update_best(in_set2, result2)

    # Use the best result for local search
    in_set = best_in_set[:]
    result = set(best_result)

    # Local search: try to increase size by removing 1 and adding 2+
    local_search_1_for_2(in_set, result, DEADLINE - 5)
    update_best(in_set, result)

    # Local search: try 1-for-1 swaps
    local_search_1_for_1_swap(in_set, result, DEADLINE - 5)
    update_best(in_set, result)

    try_add_all(in_set, result, range(1, N + 1))
    update_best(in_set, result)

    # Shake and recover
    for k in [1, 2, 3, 4]:
        max_shakes = 50
        shakes_done = 0
        while time.time() < DEADLINE - 5 and shakes_done < max_shakes:
            old_size = len(result)
            shake_and_recover(in_set, result, k, DEADLINE - 5)
            if len(result) > old_size:
                update_best(in_set, result)
            local_search_1_for_2(in_set, result, DEADLINE - 5)
            update_best(in_set, result)
            local_search_1_for_1_swap(in_set, result, DEADLINE - 5)
            update_best(in_set, result)
            try_add_all(in_set, result, range(1, N + 1))
            update_best(in_set, result)
            shakes_done += 1
            if len(result) == old_size and shakes_done >= 5:
                break

    # Final polish
    for _ in range(5):
        candidates = [y for y in range(1, N + 1) if not in_set[y]]
        candidates.sort(key=lambda x: conflict_counts[x])
        for y in candidates:
            if can_add(y, in_set):
                in_set[y] = True
                result.add(y)
        if time.time() < DEADLINE - 3:
            in_set2 = in_set[:]
            result2 = set(result)
            candidates = [y for y in range(1, N + 1) if not in_set2[y]]
            random.shuffle(candidates)
            greedy_add(in_set2, result2, candidates)
            try_add_all(in_set2, result2, range(1, N + 1))
            if len(result2) > len(result):
                in_set = in_set2
                result = result2
                update_best(in_set, result)

    # Very final local search
    if time.time() < DEADLINE - 1:
        local_search_1_for_2(in_set, result, DEADLINE - 1)
        update_best(in_set, result)
        local_search_1_for_1_swap(in_set, result, DEADLINE - 1)
        update_best(in_set, result)
        try_add_all(in_set, result, range(1, N + 1))
        update_best(in_set, result)

    return list(result)
