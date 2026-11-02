def construct():
    import math
    import random
    
    def score_of(U):
        U = sorted(set(U))
        if 0 not in U:
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
    
    random.seed(12345)
    best = [0, 1, 3, 7, 15]
    best_score = score_of(best)
    
    # Structured candidates
    structured = [
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7],
        [0, 1, 3],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 5, 12, 29],
        [0, 1, 4, 13, 40],
        [0, 1, 3, 10, 31],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 6, 12],
        [0, 1, 4, 10, 25],
        [0, 2, 6, 18],
        [0, 1, 3, 9],
        [0, 1, 4, 13],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7, 11],
        [0, 1, 3, 7, 17],
        [0, 1, 3, 7, 19],
        [0, 1, 3, 7, 21],
        [0, 1, 3, 7, 23],
    ]
    
    for s in structured:
        sc = score_of(s)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Random search
    for _ in range(20000):
        n = random.randint(4, 7)
        max_val = random.randint(10, 60)
        elems = random.sample(range(1, max_val+1), n-1)
        s = sorted([0] + elems)
        sc = score_of(s)
        if sc > best_score:
            best_score = sc
            best = s
    
    return best
