import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all C(16,4) subsets as 16-bit masks
    all_subsets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_subsets.append(mask)

    n_subsets = len(all_subsets)

    def greedy(order):
        chosen = []
        pairs = []  # list of (a, b, a&b)
        for idx in order:
            if time.time() > deadline:
                break
            s = all_subsets[idx]
            ok = True
            for a, b, ab in pairs:
                if (a & s) == ab and (b & s) == ab:
                    ok = False
                    break
            if ok:
                chosen.append(s)
                for i in range(len(chosen) - 1):
                    a = chosen[i]
                    pairs.append((a, s, a & s))
        return chosen

    # Lexicographic order
    best = greedy(list(range(n_subsets)))
    best_size = len(best)

    # Random restarts
    rng = random.Random(42)
    while time.time() < deadline - 2.0:
        order = list(range(n_subsets))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)

    return [sorted(i + 1 for i in range(16) if (mask >> i) & 1) for mask in best]
