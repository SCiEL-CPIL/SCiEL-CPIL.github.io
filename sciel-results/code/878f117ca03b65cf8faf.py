import random
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]

    best_indices = []
    best_deficit = Fraction(1, 1)

    def get_sum(indices):
        s = Fraction(0)
        for i in indices:
            s += recips[i]
        return s

    def try_update(indices):
        nonlocal best_deficit, best_indices
        if not indices:
            return
        s = get_sum(indices)
        if s >= 1:
            return
        d = 1 - s
        if d < best_deficit:
            best_deficit = d
            best_indices = list(indices)

    # Greedy initial: add in natural order (largest first)
    cur_sum = Fraction(0)
    cur_indices = []
    for i in range(n):
        if cur_sum + recips[i] < 1:
            cur_sum += recips[i]
            cur_indices.append(i)
    try_update(cur_indices)

    def local_search(indices, cur_sum):
        """Iteratively improve until local optimum. Returns (indices, sum)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_set = set(indices)
            cur_def = 1 - cur_sum

            # 1) Try adding one element
            for i in range(n):
                if i in cur_set:
                    continue
                ns = cur_sum + recips[i]
                if ns < 1:
                    nd = 1 - ns
                    if nd < cur_def:
                        indices.append(i)
                        cur_sum = ns
                        cur_set.add(i)
                        improved = True
                        break

            if improved:
                try_update(indices)
                continue

            # 2) Try swapping one for one
            for rem in list(indices):
                if time.time() >= deadline:
                    break
                s_wo = cur_sum - recips[rem]
                for add in range(n):
                    if add in cur_set and add != rem:
                        continue
                    if add == rem:
                        continue
                    ns = s_wo + recips[add]
                    if ns < 1:
                        nd = 1 - ns
                        if nd < cur_def:
                            new_idx = [x for x in indices if x != rem]
                            if add not in new_idx:
                                new_idx.append(add)
                            indices = new_idx
                            cur_sum = ns
                            cur_set = set(indices)
                            improved = True
                            break
                if improved:
                    try_update(indices)
                    break

            if improved:
                continue

            # 3) Try 2-for-1 swap: remove one, add two
            if time.time() < deadline - 3:
                for rem in list(indices):
                    if time.time() >= deadline:
                        break
                    s_wo = cur_sum - recips[rem]
                    for a in range(n):
                        if time.time() >= deadline:
                            break
                        if a in cur_set and a != rem:
                            continue
                        if a == rem:
                            continue
                        sa = s_wo + recips[a]
                        if sa >= 1:
                            continue
                        for b in range(n):
                            if b in cur_set and b != rem and b != a:
                                continue
                            if b == rem or b == a:
                                continue
                            ns = sa + recips[b]
                            if ns < 1:
                                nd = 1 - ns
                                if nd < cur_def:
                                    new_idx = [x for x in indices if x != rem]
                                    if a not in new_idx:
                                        new_idx.append(a)
                                    if b not in new_idx:
                                        new_idx.append(b)
                                    indices = new_idx
                                    cur_sum = ns
                                    cur_set = set(indices)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        try_update(indices)
                        break

            if improved:
                continue

            # 4) Try 2-for-2 swap: remove two, add two
            if time.time() < deadline - 2:
                idx_list = list(indices)
                for i1 in range(len(idx_list)):
                    if time.time() >= deadline:
                        break
                    for i2 in range(i1 + 1, len(idx_list)):
                        if time.time() >= deadline:
                            break
                        rem1 = idx_list[i1]
                        rem2 = idx_list[i2]
                        s_wo = cur_sum - recips[rem1] - recips[rem2]
                        for a in range(n):
                            if time.time() >= deadline:
                                break
                            if a in cur_set and a != rem1 and a != rem2:
                                continue
                            if a == rem1 or a == rem2:
                                continue
                            sa = s_wo + recips[a]
                            if sa >= 1:
                                continue
                            for b in range(n):
                                if b in cur_set and b != rem1 and b != rem2 and b != a:
                                    continue
                                if b == rem1 or b == rem2 or b == a:
                                    continue
                                ns = sa + recips[b]
                                if ns < 1:
                                    nd = 1 - ns
                                    if nd < cur_def:
                                        new_idx = [x for x in indices if x != rem1 and x != rem2]
                                        if a not in new_idx:
                                            new_idx.append(a)
                                        if b not in new_idx:
                                            new_idx.append(b)
                                        indices = new_idx
                                        cur_sum = ns
                                        cur_set = set(indices)
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        try_update(indices)
                        break

        return indices, cur_sum

    # Random restarts with greedy repair
    random.seed(12345)
    for restart in range(500):
        if time.time() >= deadline:
            break

        # Random subset initialization
        size = random.randint(5, 25)
        indices = random.sample(range(n), size)

        # Repair: remove largest elements until sum < 1
        cur_sum = get_sum(indices)
        if cur_sum >= 1:
            sorted_idx = sorted(indices, key=lambda x: recips[x], reverse=True)
            for i in sorted_idx:
                if cur_sum < 1:
                    break
                indices.remove(i)
                cur_sum -= recips[i]

        if not indices:
            continue

        # Run local search
        indices, cur_sum = local_search(indices, cur_sum)
        try_update(indices)

    return best_indices
