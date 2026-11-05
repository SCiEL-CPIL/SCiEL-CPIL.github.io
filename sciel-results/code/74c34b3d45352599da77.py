def construct():
    import time
    import random
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s limit

    N = 180

    # Precompute all forbidden triples: (a, b, c) with a < b < c and a*(b+c) == b*c
    # For each element, store its conflict pairs
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
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
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))

    def can_add(x, in_set):
        for (u, v) in conflict_pairs[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    def greedy_add(in_set, result, order):
        """Add elements in given order if possible."""
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                result.add(x)

    def try_add_all(in_set, result, candidates):
        """Try to add all possible candidates repeatedly until no more can be added."""
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
        """Local search: try removing one element and adding two or more."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in sorted(result):
                if time.time() > deadline:
                    break
                
                # Remove x
                in_set[x] = False
                result.discard(x)
                
                # Try to add as many as possible
                added = []
                for y in range(1, N + 1):
                    if in_set[y]:
                        continue
                    if can_add(y, in_set):
                        in_set[y] = True
                        result.add(y)
                        added.append(y)
                
                if len(added) > 1:
                    # Net gain, keep it
                    improved = True
                elif len(added) == 0:
                    # No improvement, restore x
                    in_set[x] = True
                    result.add(x)
                else:
                    # Exactly 1 added: swap. Keep the new state (might help later)
                    improved = True

    def shake_and_recover(in_set, result, k, deadline):
        """Remove k random elements, then greedily add back what fits."""
        if len(result) < k + 2:
            return False
        removed = random.sample(sorted(result), k)
        for x in removed:
            in_set[x] = False
            result.discard(x)
        
        # Greedily add back in random order
        candidates = [y for y in range(1, N + 1) if not in_set[y]]
        random.shuffle(candidates)
        greedy_add(in_set, result, candidates)
        try_add_all(in_set, result, range(1, N + 1))
        return len(result) >= len(set(removed))  # at least recovered

    # Multiple greedy strategies
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

    # Randomized restarts: shuffle the order of greedy addition
    random.seed(42)
    num_restarts = 0
    max_restarts = 200
    
    while time.time() < DEADLINE - 10 and num_restarts < max_restarts:
        num_restarts += 1
        in_set = [False] * (N + 1)
        result = set()
        
        # Randomize: pick a random subset of "seed" elements to start with
        # Bias toward larger numbers since they appear in fewer triples
        seed_size = random.randint(30, 80)
        weights = [1.0 / i for i in range(1, N + 1)]  # inverse: favor larger
        total_w = sum(weights)
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
        
        # Greedily add remaining in random order
        candidates = [i for i in range(1, N + 1) if not in_set[i]]
        random.shuffle(candidates)
        greedy_add(in_set, result, candidates)
        try_add_all(in_set, result, range(1, N + 1))
        update_best(in_set, result)
        
        # Also try: start empty, random greedy
        if time.time() < DEADLINE - 10:
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
    local_search_1_for_2(in_set, result, DEADLINE - 3)
    update_best(in_set, result)

    # Shake and recover to escape local optima
    for k in [1, 2, 3]:
        while time.time() < DEADLINE - 3:
            old_size = len(result)
            shake_and_recover(in_set, result, k, DEADLINE - 3)
            if len(result) > old_size:
                update_best(in_set, result)
            # Try local search after shake
            local_search_1_for_2(in_set, result, DEADLINE - 3)
            update_best(in_set, result)
            # If we keep getting stuck at same size, break
            if len(result) == old_size:
                break

    # Final polish: try to add any remaining in multiple orders
    for _ in range(3):
        for y in range(1, N + 1):
            if not in_set[y] and can_add(y, in_set):
                in_set[y] = True
                result.add(y)
        if time.time() < DEADLINE - 2:
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

    return list(result)
