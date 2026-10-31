import math
import itertools

def score_set(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a+b)
            diffs.add(a-b)
    size_sum = len(sums)
    size_diff = len(diffs)
    if size_sum == 0:
        return 0
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def construct():
    # Search for best small set
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Try sizes 3 to 6, max up to 40
    # To keep it fast, we can prune or just run it since it's small
    # Combinations of 3 from 0..40: 41*40*39/6 ~ 10k. Easy.
    # Combinations of 4 from 0..40: ~ 90k. Easy.
    # Combinations of 5 from 0..40: ~ 600k. Might take a second or two.
    # Combinations of 6 from 0..40: ~ 3M. Might be slow if not vectorized or pruned.
    # Let's limit max to 30 for size 6.
    
    max_val = 30
    for n in range(3, 7):
        if n == 6:
            lim = 25
        else:
            lim = max_val
            
        # Generate combinations including 0
        # Instead of combinations of range, we can iterate
        # It's easier to use itertools.combinations on range(1, lim+1) and add 0
        for comb in itertools.combinations(range(1, lim+1), n-1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set
