"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all C(16,4) subsets as 16-bit masks
    all_subsets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_subsets.append(mask)
    n_subsets = len(all_subsets)

    # Precompute intersection sizes between all pairs
    inter = [[0]*n_subsets for _ in range(n_subsets)]
    for i in range(n_subsets):
        mi = all_subsets[i]
        for j in range(i+1, n_subsets):
            mj = all_subsets[j]
            c = (mi & mj).bit_count()
            inter[i][j] = c
            inter[j][i] = c

    def greedy(order):
        chosen = []
        chosen_set = set()
        # pair_by_first[a][v] = list of b's such that a,b in chosen and inter[a][b] == v
        pair_by_first = [dict() for _ in range(n_subsets)]
        
        for idx in order:
            if time.time() > deadline:
                break
            s = idx
            if s in chosen_set:
                continue
            ok = True
            for a in chosen:
                ia = inter[a][s]
                if ia in pair_by_first[a]:
                    for b in pair_by_first[a][ia]:
                        if inter[b][s] == ia:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                chosen.append(s)
                chosen_set.add(s)
                for a in chosen[:-1]:
                    ia = inter[a][s]
                    if ia not in pair_by_first[a]:
                        pair_by_first[a][ia] = []
                    pair_by_first[a][ia].append(s)
                    if ia not in pair_by_first[s]:
                        pair_by_first[s][ia] = []
                    pair_by_first[s][ia].append(a)
        return chosen

    best = []
    best_size = 0

    # Lexicographic
    result = greedy(list(range(n_subsets)))
    if len(result) > best_size:
        best = result
        best_size = len(result)

    # Random restarts
    rng = random.Random(42)
    while time.time() < deadline - 3.0:
        order = list(range(n_subsets))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)

    return [sorted(i + 1 for i in range(16) if (all_subsets[idx] >> i) & 1) for idx in best]
# EVOLVE-BLOCK-END
