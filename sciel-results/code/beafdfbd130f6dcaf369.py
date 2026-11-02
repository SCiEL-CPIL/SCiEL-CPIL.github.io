def construct():
    import math
    import itertools

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

    best_score = 0
    best_candidate = [0]
    
    # Search small sets
    # Size 3
    for a in range(1, 10):
        for b in range(a+1, 20):
            U = [0, a, b]
            sc = score_of(U)
            if sc > best_score:
                best_score = sc
                best_candidate = U[:]
                
    # Size 4
    for a in range(1, 10):
        for b in range(a+1, 20):
            for c in range(b+1, 30):
                U = [0, a, b, c]
                sc = score_of(U)
                if sc > best_score:
                    best_score = sc
                    best_candidate = U[:]

    return best_candidate
