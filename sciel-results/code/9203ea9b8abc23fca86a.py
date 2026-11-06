"""Optimized: precomputed AP structures + simulated annealing local search."""
import time
import random
import math

N = 256
DEADLINE = 80.0


def construct():
    start = time.time()
    random.seed(42)

    # Precompute 4-AP conflicts for each x
    four_ap_conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            four_ap_conflicts[a].append((b, c, e))
            four_ap_conflicts[b].append((a, c, e))
            four_ap_conflicts[c].append((a, b, e))
            four_ap_conflicts[e].append((a, b, c))

    # Precompute 3-AP partners
    three_ap_first = [[] for _ in range(N + 1)]
    three_ap_middle = [[] for _ in range(N + 1)]
    three_ap_last = [[] for _ in range(N + 1)]

    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            three_ap_first[a].append((b, c))
            three_ap_middle[b].append((a, c))
            three_ap_last[c].append((a, b))

    def has_4ap(mask, x):
        for (a, b, c) in four_ap_conflicts[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1) and ((mask >> (c - 1)) & 1):
                return True
        return False

    def delta_add(mask, x):
        cnt = 0
        for (a, b) in three_ap_first[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
                cnt += 1
        for (a, b) in three_ap_middle[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
                cnt += 1
        for (a, b) in three_ap_last[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
                cnt += 1
        return cnt

    def count_3aps(mask):
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                m = mask & ~(1 << (x - 1))
                total += delta_add(m, x)
        return total

    def greedy_seed(seed_val):
        rng = random.Random(seed_val)
        mask = 0
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if not has_4ap(mask, x):
                mask |= (1 << (x - 1))
        return mask

    # Try multiple greedy seeds
    best_mask = 0
    best_3ap = -1
    for seed_val in range(30):
        if time.time() - start > DEADLINE - 10:
            break
        m = greedy_seed(seed_val)
        c = count_3aps(m)
        if c > best_3ap:
            best_3ap = c
            best_mask = m

    # Local search with simulated annealing
    mask = best_mask
    current_3ap = best_3ap
    best_mask = mask
    best_3ap = current_3ap

    T = 3.0
    T_min = 0.05
    cooling = 0.998

    while time.time() - start < DEADLINE:
        improved = False
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]

        if not elems or not missing:
            break

        random.shuffle(elems)
        random.shuffle(missing)

        for rem in elems:
            if time.time() - start > DEADLINE:
                break
            for add in missing:
                if time.time() - start > DEADLINE:
                    break
                trial_mask = (mask & ~(1 << (rem - 1))) | (1 << (add - 1))
                if has_4ap(trial_mask & ~(1 << (add - 1)), add):
                    continue
                delta = delta_add(trial_mask & ~(1 << (add - 1)), add) - delta_add(mask & ~(1 << (rem - 1)), rem)
                if delta > 0:
                    mask = trial_mask
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_mask = mask
                    improved = True
                    break
                else:
                    if T > T_min and delta > -15:
                        if random.random() < math.exp(delta / T):
                            mask = trial_mask
                            current_3ap += delta
                            if current_3ap > best_3ap:
                                best_3ap = current_3ap
                                best_mask = mask
                            improved = True
                            break
            if improved:
                break

        if not improved:
            found = False
            for add in missing:
                if time.time() - start > DEADLINE:
                    break
                if has_4ap(mask, add):
                    continue
                delta = delta_add(mask, add)
                if delta > 0:
                    mask |= (1 << (add - 1))
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_mask = mask
                    found = True
                    break

        T *= cooling
        if T < T_min:
            T = T_min
            if elems:
                rem = random.choice(elems)
                mask &= ~(1 << (rem - 1))
                current_3ap -= delta_add(mask | (1 << (rem - 1)), rem)

    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
