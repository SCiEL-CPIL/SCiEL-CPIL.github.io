import time

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}
_DEADLINE = 85.0

def construct():
    t_start = time.time()
    N = P['n']
    MAXSIZE = P['maximum_size']

    def try_build(order):
        sums = [0] * (MAXSIZE + 1)
        sums[0] = 1
        chosen = []
        for c in order:
            if time.time() > _DEADLINE:
                break
            k_max = len(chosen)
            # For each k from k_max down to 0, the new sums at level k would be:
            #   sums[k] | (sums[k-1] << c)  [if k>=1]
            # New bits added at level k: (sums[k-1] << c) & ~sums[k]
            # These new bits must not collide with any sums[j] for j != k.
            # Also, the shifted bits from level k-1 must not collide with sums[j] for j != k-1, j != k.
            # Simplest correct check: compute all new_sums and verify pairwise disjointness.
            # But we can shortcut: since old sums are already disjoint across levels,
            # we only need to check that (sums[k-1] << c) doesn't hit sums[j] for j != k-1,
            # AND that the new bits at level k don't hit other levels.
            # Actually, let's just do the full check but efficiently.
            
            valid = True
            # Precompute shifted values
            shifted = [0] * (k_max + 1)
            for k in range(1, k_max + 1):
                shifted[k] = sums[k - 1] << c
            
            # For each level k, the new bits are shifted[k] & ~sums[k]
            # These must not intersect with sums[j] for j != k
            # Also shifted[k] must not intersect with sums[j] for j != k-1 (since shifted[k] adds to level k,
            # but the bits come from level k-1 shifted, they could collide with any level j)
            # Full check: for each k, new_bits_k = shifted[k] & ~sums[k]
            #   must satisfy new_bits_k & sums[j] == 0 for all j != k
            #   AND shifted[k] & sums[j] == 0 for all j not in {k-1, k} (since sums[k-1] is already disjoint from sums[j] for j!=k-1,
            #   but shifted version might not be)
            # Wait, this is getting complicated. Let me just compute new_sums and check.
            
            new_sums = list(sums)
            for k in range(k_max, -1, -1):
                if k >= 1:
                    new_sums[k] = sums[k] | (sums[k-1] << c)
                # else new_sums[0] = sums[0] (unchanged)
            
            # Check pairwise disjointness of new_sums
            for k in range(k_max + 1):
                if new_sums[k] == 0:
                    continue
                for j in range(k + 1, k_max + 1):
                    if new_sums[j] == 0:
                        continue
                    if (new_sums[k] & new_sums[j]) != 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                sums = new_sums
                chosen.append(c)
                if len(chosen) >= MAXSIZE:
                    break
        return chosen

    # Strategy 1: Descending order
    result = try_build(range(N, 0, -1))
    best = result

    if time.time() > _DEADLINE:
        return best

    # Strategy 2: Spaced descending
    for step in [2, 3, 4, 5, 6]:
        if time.time() > _DEADLINE:
            break
        order = list(range(N - N % step, 0, -step))
        r = try_build(order)
        if len(r) > len(best):
            best = r

    # Strategy 3: Ascending
    if time.time() > _DEADLINE:
        return best
    r = try_build(range(1, N + 1))
    if len(r) > len(best):
        best = r

    # Strategy 4: Interleaved
    if time.time() > _DEADLINE:
        return best
    order = []
    lo, hi = 1, N
    while lo <= hi:
        order.append(hi)
        if lo != hi:
            order.append(lo)
        lo += 1
        hi -= 1
    r = try_build(order)
    if len(r) > len(best):
        best = r

    # Strategy 5: Exponential spacing then fill
    if time.time() > _DEADLINE:
        return best
    exp_order = []
    v = 1
    while v <= N:
        exp_order.append(v)
        v = v * 2 + 1
    seen = set(exp_order)
    rest = [x for x in range(N, 0, -1) if x not in seen]
    order = exp_order + rest
    r = try_build(order)
    if len(r) > len(best):
        best = r

    # Strategy 6: Randomized restarts
    import random
    if time.time() > _DEADLINE:
        return best
    rng = random.Random(12345)
    for attempt in range(100):
        if time.time() > _DEADLINE:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # Bias toward larger numbers
        order.sort(key=lambda x: -x + rng.uniform(-30, 30))
        r = try_build(order)
        if len(r) > len(best):
            best = r

    best = [int(x) for x in best]
    return best
# EVOLVE-BLOCK-END
