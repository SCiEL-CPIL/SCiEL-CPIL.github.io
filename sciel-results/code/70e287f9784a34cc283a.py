"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0  # safety margin before 90s kill

    N = 256
    ALL = (1 << N) - 1

    # Precompute conflict_mask[i]: bitmask of j such that (i+1, j+1) conflict
    # Use the algebraic characterization: (a+b)|ab iff (m+n)|d where a=d*m, b=d*n, gcd(m,n)=1
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        conflict_mask[i] = m

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]

    # base = all odds (mutually compatible: odd*odd is odd, odd+odd is even, so (odd+odd) can't divide odd)
    base_mask = 0
    for i in odds_indices:
        base_mask |= (1 << i)

    def popcount(x):
        return x.bit_count()

    def greedy_add(start_mask, order, deadline):
        cur = start_mask
        for e in order:
            if time.time() > deadline:
                break
            if (cur >> e) & 1:
                continue
            if (conflict_mask[e] & cur) == 0:
                cur |= (1 << e)
        return cur

    def fast_repair(mask, deadline):
        """Single pass greedy add."""
        cur = mask
        not_in = ALL & ~cur
        while not_in:
            if time.time() > deadline:
                break
            lsb = not_in & (-not_in)
            e = lsb.bit_length() - 1
            not_in ^= lsb
            if (conflict_mask[e] & cur) == 0:
                cur |= lsb
        return cur

    def local_search(start_mask, deadline, max_iters=300000, seed=12345):
        best_mask = start_mask
        best_size = popcount(start_mask)
        cur = start_mask
        cur_size = best_size
        rng = random.Random(seed)
        no_improve = 0
        patience = 50000
        T = 1.0
        T_min = 0.1
        cooling = 0.99999

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Simulated Annealing cooling
            T = max(T_min, T * cooling)

            r = rng.random()
            new_cur = cur
            new_size = cur_size
            accepted = False

            if r < 0.5:
                # ADD a random compatible element
                not_in = ALL & ~cur
                if not_in == 0:
                    continue
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                if (conflict_mask[e] & cur) == 0:
                    new_cur = cur | (1 << e)
                    new_size = cur_size + 1
                    accepted = True
            elif r < 0.8:
                # 1-SWAP: remove a conflicting element of e, add e
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                conflicts = conflict_mask[e] & cur
                if conflicts == 0:
                    new_cur = cur | (1 << e)
                    new_size = cur_size + 1
                    accepted = True
                else:
                    # pick a random conflicting element to remove
                    k = rng.randrange(popcount(conflicts))
                    c = None
                    cc = conflicts
                    for _ in range(k + 1):
                        lsb = cc & (-cc)
                        c = lsb.bit_length() - 1
                        cc ^= lsb
                    new_cur = (cur & ~(1 << c)) | (1 << e)
                    # Repair
                    new_cur = fast_repair(new_cur, deadline)
                    new_size = popcount(new_cur)
                    # Accept if better or equal, or worse with prob exp(-delta/T)
                    if new_size >= cur_size:
                        accepted = True
                    else:
                        delta = cur_size - new_size
                        if rng.random() < math.exp(-delta / T):
                            accepted = True
            elif r < 0.9:
                # 2-SWAP: remove one, add one, repair
                e1 = rng.randrange(N)
                if (cur >> e1) & 1:
                    continue
                conflicts1 = conflict_mask[e1] & cur
                if conflicts1 == 0:
                    new_cur = cur | (1 << e1)
                    new_size = cur_size + 1
                    accepted = True
                else:
                    k1 = rng.randrange(popcount(conflicts1))
                    c1 = None
                    cc = conflicts1
                    for _ in range(k1 + 1):
                        lsb = cc & (-cc)
                        c1 = lsb.bit_length() - 1
                        cc ^= lsb
                    trial = cur & ~(1 << c1)
                    if (conflict_mask[e1] & trial) == 0:
                        trial |= (1 << e1)
                    # Try to add a second element to compensate if size dropped
                    # Or just repair
                    new_cur = fast_repair(trial, deadline)
                    new_size = popcount(new_cur)
                    if new_size >= cur_size:
                        accepted = True
                    else:
                        delta = cur_size - new_size
                        if rng.random() < math.exp(-delta / T):
                            accepted = True
            else:
                # 2-REMOVE + repair
                drop = 1 if rng.random() < 0.5 else 2
                trial = cur
                for _ in range(drop):
                    if trial == 0:
                        break
                    e = rng.randrange(N)
                    if (trial >> e) & 1:
                        trial &= ~(1 << e)
                new_cur = fast_repair(trial, deadline)
                new_size = popcount(new_cur)
                if new_size >= cur_size:
                    accepted = True
                else:
                    delta = cur_size - new_size
                    if rng.random() < math.exp(-delta / T):
                        accepted = True

            if accepted:
                cur = new_cur
                cur_size = new_size
                no_improve = 0
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur
                    T = 1.0 # Reset temperature on improvement
            
            if _ > 0 and _ % 1000 == 0:
                if cur_size <= best_size:
                    no_improve += 1000
                else:
                    no_improve = 0
            if no_improve > patience:
                break

        return best_mask

    # ---- Multi-start diverse greedy ----
    best_overall = base_mask
    best_size = popcount(base_mask)

    def consider(mask):
        nonlocal best_overall, best_size
        sz = popcount(mask)
        if sz > best_size:
            best_overall = mask
            best_size = sz

    # 1) natural order
    consider(fast_repair(greedy_add(base_mask, evens_indices, DEADLINE), DEADLINE))
    # 2) reverse order
    consider(fast_repair(greedy_add(base_mask, list(reversed(evens_indices)), DEADLINE), DEADLINE))
    # 3) ascending by conflict-degree with odds (low-degree first)
    deg = [(popcount(conflict_mask[e] & base_mask), e) for e in evens_indices]
    deg.sort()
    consider(fast_repair(greedy_add(base_mask, [e for _, e in deg], DEADLINE), DEADLINE))
    # 4) descending degree
    deg.sort(key=lambda x: -x[0])
    consider(fast_repair(greedy_add(base_mask, [e for _, e in deg], DEADLINE), DEADLINE))

    # 5) many random orderings
    for attempt in range(100):
        if time.time() > DEADLINE - 30:
            break
        order = evens_indices[:]
        random.shuffle(order)
        consider(fast_repair(greedy_add(base_mask, order, DEADLINE), DEADLINE))

    # ---- Local search from best, with several restarts from diverse starts ----
    if time.time() < DEADLINE - 20:
        improved = local_search(best_overall, DEADLINE, max_iters=600000, seed=12345)
        consider(improved)

    # A couple of extra local-search restarts from other good starts if time allows
    extra_starts = []
    for attempt in range(10):
        if time.time() > DEADLINE - 12:
            break
        order = evens_indices[:]
        random.shuffle(order)
        m = fast_repair(greedy_add(base_mask, order, DEADLINE), DEADLINE)
        if popcount(m) >= best_size - 2:
            extra_starts.append(m)

    for idx, m in enumerate(extra_starts):
        if time.time() > DEADLINE - 5:
            break
        improved = local_search(m, DEADLINE, max_iters=80000, seed=12345 + idx * 7919)
        consider(improved)

    # Final local search polish on the best
    if time.time() < DEADLINE - 2:
        improved = local_search(best_overall, DEADLINE, max_iters=150000, seed=99999)
        consider(improved)

    result = sorted([i + 1 for i in range(N) if (best_overall >> i) & 1])

    # Final verification; fall back to odds if any conflict slips through
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                return sorted([i + 1 for i in odds_indices])

    return result
# EVOLVE-BLOCK-END
