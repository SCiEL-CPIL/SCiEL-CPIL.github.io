def construct():
    import time
    import random
    from math import lcm
    from itertools import combinations

    deadline = time.time() + 80  # 80s safety margin out of 90s

    def is_bad_triple(a, b, c):
        return lcm(a, b) == lcm(a, c) == lcm(b, c)

    def is_valid(set_list):
        s = set(set_list)
        for a, b, c in combinations(s, 3):
            if is_bad_triple(a, b, c):
                return False
        return True

    # Precompute lcm table
    lcm_table = {}
    for i in range(1, 129):
        for j in range(i, 129):
            lcm_table[(i, j)] = lcm(i, j)
    # Make it symmetric
    for i in range(1, 129):
        for j in range(1, 129):
            if (i, j) not in lcm_table:
                lcm_table[(i, j)] = lcm_table[(j, i)]

    def check_add(current_set, x):
        """Check if adding x to current_set creates a bad triple."""
        for a, b in combinations(current_set, 2):
            if lcm_table[(a, b)] == lcm_table[(a, x)] == lcm_table[(b, x)]:
                return False
        return True

    # Greedy initialization with random order
    best_set = set()
    for _ in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        current = set()
        for x in candidates:
            if check_add(current, x):
                current.add(x)
        if len(current) > len(best_set):
            best_set = set(current)

    # Local search: try swaps and additions
    current = set(best_set)
    best_size = len(current)

    # Precompute for current set
    def try_add_to_current(current, x):
        for a, b in combinations(current, 2):
            if lcm_table[(a, b)] == lcm_table[(a, x)] == lcm_table[(b, x)]:
                return False
        return True

    # Iterative improvement
    for iteration in range(2000):
        if time.time() > deadline:
            break

        # Try to add a random element
        candidates = [x for x in range(1, 129) if x not in current]
        if candidates:
            x = random.choice(candidates)
            if try_add_to_current(current, x):
                current.add(x)
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
                continue

        # Try a swap: remove one, add one or two
        if candidates:
            x = random.choice(candidates)
            # Find elements that block x
            blockers = set()
            for a, b in combinations(current, 2):
                if lcm_table[(a, b)] == lcm_table[(a, x)] == lcm_table[(b, x)]:
                    blockers.add(a)
                    blockers.add(b)
            if blockers:
                to_remove = random.choice(list(blockers))
                current.remove(to_remove)
                if try_add_to_current(current, x):
                    current.add(x)
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    else:
                        current.remove(x)
                        current.add(to_remove)
                else:
                    current.add(to_remove)

    return sorted(best_set)
