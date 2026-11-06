"""Fast 4-AP-free set construction with 3-AP maximization."""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin under 60s kill


def construct():
    start = time.time()
    random.seed(12345)

    # Precompute for each x: bitmask of all y such that {x,y} is part of a 4-AP
    # conflict[x] is a 256-bit int where bit y is set if x and y are in some 4-AP
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                for d in range(c + 1, N + 1):
                    # Check if a,b,c,d form a 4-AP
                    if (b - a) == (c - b) == (d - c):
                        # All pairs in this 4-AP conflict
                        for i in range(4):
                            for j in range(i + 1, 4):
                                x, y = (a, b, c, d)[i], (a, b, c, d)[j]
                                conflict[x] |= (1 << (y - 1))
                                conflict[y] |= (1 << (x - 1))

    def has_4ap_with(mask, x):
        """Check if adding x to mask creates a 4-AP. Fast: just check conflict."""
        if (mask >> (x - 1)) & 1:
            return False  # x already in mask
        return (mask & conflict[x]) != 0

    def count_3aps_in_mask(mask):
        """Count 3-APs in mask using direct enumeration."""
        cnt = 0
        for a in range(1, N + 1):
            if not ((mask >> (a - 1)) & 1):
                continue
            for d in range(1, (N - a) // 2 + 1):
                b = a + d
                c = a + 2 * d
                if (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1:
                    cnt += 1
        return cnt

    def greedy(order):
        """Greedy: add elements in order if no 4-AP formed."""
        m = 0
        for x in order:
            if not has_4ap_with(m, x):
                m |= (1 << (x - 1))
        return m

    def local_search(mask, budget):
        """Fast local search: try swaps and additions."""
        while time.time() - start < budget:
            improved = False
            elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
            random.shuffle(elems)
            random.shuffle(missing)

            # Try single additions
            for x in missing:
                if time.time() - start > budget:
                    break
                if not has_4ap_with(mask, x):
                    # Check if this adds 3-APs
                    old_cnt = count_3aps_in_mask(mask)
                    new_mask = mask | (1 << (x - 1))
                    new_cnt = count_3aps_in_mask(new_mask)
                    if new_cnt > old_cnt:
                        mask = new_mask
                        improved = True
                        break

            if improved:
                continue

            # Try swaps: remove one, add another
            for rem in elems:
                if time.time() - start > budget:
                    break
                mask_no_rem = mask & ~(1 << (rem - 1))
                old_cnt = count_3aps_in_mask(mask_no_rem)
                for add in missing:
                    if time.time() - start > budget:
                        break
                    if not has_4ap_with(mask_no_rem, add):
                        new_mask = mask_no_rem | (1 << (add - 1))
                        new_cnt = count_3aps_in_mask(new_mask)
                        if new_cnt > old_cnt:
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                # Perturbation: remove 3 random elements
                if len(elems) >= 3:
                    to_remove = random.sample(elems, 3)
                    for x in to_remove:
                        mask &= ~(1 << (x - 1))

        return mask

    # Multi-start
    best_mask = 0
    best_3ap = 0

    # Start 1: natural order
    m = greedy(list(range(1, N + 1)))
    c = count_3aps_in_mask(m)
    if c > best_3ap:
        best_3ap, best_mask = c, m

    # Start 2: reverse order
    m = greedy(list(range(N, 0, -1)))
    c = count_3aps_in_mask(m)
    if c > best_3ap:
        best_3ap, best_mask = c, m

    # Start 3: random order
    random.seed(42)
    order = list(range(1, N + 1))
    random.shuffle(order)
    m = greedy(order)
    c = count_3aps_in_mask(m)
    if c > best_3ap:
        best_3ap, best_mask = c, m

    # Run local search on best
    if time.time() - start < DEADLINE - 5:
        m = local_search(best_mask, DEADLINE - 3)
        c = count_3aps_in_mask(m)
        if c > best_3ap:
            best_3ap, best_mask = c, m

    # More restarts with perturbation
    for restart in range(5):
        if time.time() - start > DEADLINE - 5:
            break
        m = best_mask
        elems = [x for x in range(1, N + 1) if (m >> (x - 1)) & 1]
        if len(elems) >= 4:
            to_remove = random.sample(elems, random.randint(3, 6))
            for x in to_remove:
                m &= ~(1 << (x - 1))
        m = local_search(m, min(time.time() - start + 5, DEADLINE - 3))
        c = count_3aps_in_mask(m)
        if c > best_3ap:
            best_3ap, best_mask = c, m

    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
