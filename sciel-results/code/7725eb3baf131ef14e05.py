# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 83  # 83s deadline, 7s safety margin before 90s kill

    # Precompute all 4-element subsets of [1,16] as 16-bit masks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)

    n_total = len(all_sets)  # C(16,4) = 1820

    # Precompute pairwise intersections for all pairs of masks
    # inter[i][j] = all_sets[i] & all_sets[j]
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_sets[i]
        for j in range(i + 1, n_total):
            v = mi & all_sets[j]
            inter[i][j] = v
            inter[j][i] = v

    best_size = 0
    best_family = []

    def greedy_insertion(order, family, in_family):
        """Greedy insert sets in given order. family is list of indices, in_family is set of indices."""
        for s_idx in order:
            if in_family is not None and s_idx in in_family:
                continue
            s = all_sets[s_idx]
            is_valid = True
            nf = len(family)
            for i in range(nf):
                a = all_sets[family[i]]
                for j in range(i + 1, nf):
                    b = all_sets[family[j]]
                    ab = inter[family[i]][family[j]]
                    if (a & s) == ab and (b & s) == ab:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                family.append(s_idx)
                if in_family is not None:
                    in_family.add(s_idx)

    # Try multiple strategies
    for attempt in range(5000):
        if time.time() > deadline:
            break

        if attempt == 0:
            order = list(range(n_total))
        elif attempt == 1:
            order = list(range(n_total))
            random.shuffle(order)
        else:
            order = list(range(n_total))
            random.shuffle(order)

        family = []
        greedy_insertion(order, family, None)

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]

        if best_size >= 250:
            break

    # Polish: try to add any remaining set to best_family
    if best_size < n_total and time.time() < deadline:
        in_family = set(best_family)
        for s_idx in range(n_total):
            if s_idx in in_family:
                continue
            s = all_sets[s_idx]
            is_valid = True
            nf = len(best_family)
            for i in range(nf):
                a = all_sets[best_family[i]]
                for j in range(i + 1, nf):
                    b = all_sets[best_family[j]]
                    ab = inter[best_family[i]][best_family[j]]
                    if (a & s) == ab and (b & s) == ab:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                best_family.append(s_idx)
                in_family.add(s_idx)
                best_size = len(best_family)

    # Convert indices back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for k in range(16):
            if mask & (1 << k):
                elems.append(k + 1)
        result.append(elems)

    return result
# EVOLVE-BLOCK-END
