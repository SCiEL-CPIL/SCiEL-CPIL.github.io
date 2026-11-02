def construct():
    import math
    import itertools
    import time

    def score_of(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        diff_set = set()
        sum_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
                sum_set.add(a + b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0 or n_diff == 0:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1
        return 1 + math.log(ratio) / denom

    best_candidate = [0, 1, 4, 6]
    best_score = score_of(best_candidate)
    
    start_time = time.time()
    time_limit = 115

    # Systematic search for small sets
    # Sizes 3 to 6
    # Max values adjusted to keep search space small
    search_params = [
        (3, 20),
        (4, 30),
        (5, 25),
        (6, 20),
    ]
    
    for n, max_val in search_params:
        if time.time() - start_time > time_limit:
            break
        # Generate all combinations of size n-1 from 1..max_val
        # The set will be [0] + comb
        for comb in itertools.combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(comb)
            sc = score_of(U)
            if sc > best_score:
                best_score = sc
                best_candidate = U[:]
            if time.time() - start_time > time_limit:
                break

    return best_candidate
