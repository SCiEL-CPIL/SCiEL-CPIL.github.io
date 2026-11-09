"""Rewritten: bit-parallel conflict-mask local search for the 3-uniform hypergraph."""
import time
import random

N = 128
MAXV = 1 << (N + 1)  # mask over bits 1..128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def time_left():
        return deadline - time.time()

    # ---- Precompute conflict masks and partner masks (bit-parallel) ----
    conflict = [0] * (N + 1)
    partner = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = a + b
            if ab > N:
                continue
            for c in range(b + 1, N + 1):
                ac = a + c
                bc = b + c
                if ac > N or bc > N:
                    continue
                # {a,b,c} is a forbidden triple
                conflict[a] |= (1 << b) | (1 << c)
                conflict[b] |= (1 << a) | (1 << c)
                conflict[c] |= (1 << a) | (1 << b)
                partner[a] |= (1 << b) | (1 << c)
                partner[b] |= (1 << a) | (1 << c)
                partner[c] |= (1 << a) | (1 << b)

    def can_add(mask, x):
        cm = conflict[x] & mask
        return (cm & (cm - 1)) == 0

    def is_valid(mask):
        elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
        k = len(elems)
        for i in range(k):
            a = elems[i]
            for j in range(i + 1, k):
                b = elems[j]
                if ((partner[a] >> b) & 1) and ((partner[a] & mask) >> 1) and False:
                    pass
                # partner[a] has b set iff exists z completing triple with a,b
                if ((partner[a] >> b) & 1) and ((partner[a] & mask) and True):
                    # need a third distinct member z in mask
                    rest = partner[a] & mask
                    # rest includes b; need another bit
                    if rest & (rest - 1) & ~(1 << b):
                        return False
        return True

    def popcount(mask):
        return bin(mask).count('1')

    # ---- Initial seed: upper block {65..128} (all pair sums > 128, trivially valid) ----
    base_mask = 0
    for i in range(65, N + 1):
        base_mask |= (1 << i)

    best_mask = base_mask
    best_size = 64

    # ---- Greedy fill (fast O(1) checks) ----
    def greedy_fill(mask, order):
        for x in order:
            if (mask >> x) & 1:
                continue
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def greedy_from(mask, rng=None):
        order = list(range(1, N + 1))
        if rng is not None:
            rng.shuffle(order)
        else:
            # deterministic: try small numbers first (they interact more, fill first)
            pass
        return greedy_fill(mask, order)

    # deterministic greedy on base
    m = greedy_from(base_mask)
    if popcount(m) > best_size:
        best_mask, best_size = m, popcount(m)

    # ---- Local search with guided swaps ----
    def local_search(mask, rng, steps_budget):
        mask = greedy_from(mask, rng)
        size = popcount(mask)
        for _ in range(steps_budget):
            if time_left() < 4.0:
                break
            elems_in = [i for i in range(1, N + 1) if (mask >> i) & 1]
            elems_out = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
            if not elems_out:
                break
            r = rng.choice(elems_in)
            a = rng.choice(elems_out)
            temp = mask & ~(1 << r)
            if can_add(temp, a):
                new = temp | (1 << a)
                new = greedy_from(new, rng)
                ns = popcount(new)
                if ns > size:
                    mask, size = new, ns
        return mask, size

    rng = random.Random(12345)
    # Several guided local-search passes
    for it in range(4000):
        if time_left() < 6.0:
            break
        m, s = local_search(best_mask, rng, 30)
        if s > best_size:
            best_mask, best_size = m, s
        elif time_left() < 30.0:
            break

    # ---- Multi-restart: random seeds + greedy + local search ----
    def random_seed():
        # keep upper block plus a random subset of lower block
        m = base_mask
        for i in range(1, 65):
            if rng.random() < 0.5:
                m |= (1 << i)
        return m

    for _ in range(3000):
        if time_left() < 8.0:
            break
        m = random_seed()
        m, s = local_search(m, rng, 25)
        if s > best_size:
            best_mask, best_size = m, s

    # ---- Final greedy polish on best ----
    m = greedy_from(best_mask)
    if popcount(m) > best_size:
        best_mask, best_size = m, popcount(m)

    # ---- Final exact verification (defensive) ----
    if not is_valid(best_mask):
        # fall back to the trivially valid upper block
        best_mask = base_mask
        best_size = 64

    result = [int(i) for i in range(1, N + 1) if (best_mask >> i) & 1]
    return result
