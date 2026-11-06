"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0  # safety margin before 90s kill

    N = 256
    ALL = (1 << N) - 1

    # Precompute conflict_mask[i]: bitmask of j such that (i+1,j+1) conflict
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

    # base = all odds (mutually compatible: odd*odd odd, odd+odd even)
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

    def greedy_complete(mask, deadline, max_rounds=200):
        """Repeatedly scan all candidates and add any that fit, until no change."""
        cur = mask
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            changed = False
            not_in = ALL & ~cur
            while not_in:
                if time.time() > deadline:
                    break
                lsb = not_in & (-not_in)
                e = lsb.bit_length() - 1
                not_in ^= lsb
                if (conflict_mask[e] & cur) == 0:
                    cur |= lsb
                    changed = True
            if not changed:
                break
        return cur

    def greedy_min_degree(start_mask, deadline):
        """Add elements in order of minimum conflict degree with current set."""
        cur = start_mask
        remaining = set(range(N))
        remaining.discard(i for i in range(N) if (cur >> i) & 1)
        
        while remaining:
            if time.time() > deadline:
                break
            best_e = None
            best_deg = float('inf')
            for e in remaining:
                deg = popcount(conflict_mask[e] & cur)
                if deg < best_deg:
                    best_deg = deg
                    best_e = e
                if deg == 0:
                    break
            if best_e is None:
                break
            if best_deg == 0:
                cur |= (1 << best_e)
                remaining.discard(best_e)
            else:
                break
        return cur

    def greedy_max_degree(start_mask, deadline):
        """Add elements in order of maximum conflict degree (most constrained first)."""
        cur = start_mask
        remaining = set(range(N))
        remaining.discard(i for i in range(N) if (cur >> i) & 1)
        
        while remaining:
            if time.time() > deadline:
                break
            best_e = None
            best_deg = -1
            for e in remaining:
                deg = popcount(conflict_mask[e] & cur)
                if deg > best_deg:
                    best_deg = deg
                    best_e = e
            if best_e is None:
                break
            if best_deg == 0:
                cur |= (1 << best_e)
                remaining.discard(best_e)
            else:
                break
        return cur

    def greedy_random(start_mask, deadline, seed):
        """Add elements in random order."""
        rng = random.Random(seed)
        cur = start_mask
        remaining = list(range(N))
        remaining = [i for i in remaining if not ((cur >> i) & 1)]
        rng.shuffle(remaining)
        for e in remaining:
            if time.time() > deadline:
                break
            if (conflict_mask[e] & cur) == 0:
                cur |= (1 << e)
        return cur

    def local_search(start_mask, deadline, max_iters=200000, seed=12345):
        best_mask = start_mask
        best_size = popcount(start_mask)
        cur = start_mask
        cur_size = best_size
        rng = random.Random(seed)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            r = rng.random()
            if r < 0.50:
                # ADD a random compatible element
                not_in = ALL & ~cur
                if not_in == 0:
                    continue
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                if (conflict_mask[e] & cur) == 0:
                    cur |= (1 << e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            elif r < 0.82:
                # 1-SWAP: remove the LEAST-DAMAGING conflicting element of e, add e
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                conflicts = conflict_mask[e] & cur
                if conflicts == 0:
                    cur |= (1 << e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
                    continue
                # pick the conflicting element whose removal frees the most future adds
                best_c = None
                best_gain = -1
                cc = conflicts
                while cc:
                    lsb = cc & (-cc)
                    c = lsb.bit_length() - 1
                    cc ^= lsb
                    # gain: how many non-conflicting-with-e elements c was blocking
                    # i.e., elements x not in cur, x != e, x not conflicting with e,
                    # but x conflicting with c
                    freed = (~conflict_mask[e]) & ALL & ~(1 << e) & ~(1 << c) & conflict_mask[c] & ~cur
                    gain = popcount(freed)
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                new_mask = (cur & ~(1 << best_c)) | (1 << e)
                if (conflict_mask[e] & (new_mask & ~(1 << e))) == 0:
                    cur = new_mask
                    cur = greedy_complete(cur, deadline, max_rounds=2)
                    cur_size = popcount(cur)
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            elif r < 0.95:
                # 2-REMOVE + complete: drop 1-2 random elements, then greedy complete
                drop = 1 if rng.random() < 0.5 else 2
                trial = cur
                for _ in range(drop):
                    if trial == 0:
                        break
                    e = rng.randrange(N)
                    if (trial >> e) & 1:
                        trial &= ~(1 << e)
                trial = greedy_complete(trial, deadline, max_rounds=4)
                ts = popcount(trial)
                if ts > cur_size:
                    cur = trial
                    cur_size = ts
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            else:
                # 2-SWAP: remove two conflicting elements, add two new ones
                e1 = rng.randrange(N)
                if (cur >> e1) & 1:
                    continue
                conflicts1 = conflict_mask[e1] & cur
                if conflicts1 == 0:
                    cur |= (1 << e1)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
                    continue
                k = rng.randrange(popcount(conflicts1))
                c1 = None
                cc = conflicts1
                for _ in range(k + 1):
                    lsb = cc & (-cc)
                    c1 = lsb.bit_length() - 1
                    cc ^= lsb
                trial = cur & ~(1 << c1)
                if (conflict_mask[e1] & trial) == 0:
                    trial |= (1 << e1)
                e2 = rng.randrange(N)
                if not ((trial >> e2) & 1):
                    if (conflict_mask[e2] & trial) == 0:
                        trial |= (1 << e2)
                trial = greedy_complete(trial, deadline, max_rounds=3)
                ts = popcount(trial)
                if ts > cur_size:
                    cur = trial
                    cur_size = ts
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur

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
    consider(greedy_complete(greedy_add(base_mask, evens_indices, DEADLINE), DEADLINE))
    # 2) reverse order
    consider(greedy_complete(greedy_add(base_mask, list(reversed(evens_indices)), DEADLINE), DEADLINE))
    # 3) ascending by conflict-degree with odds (low-degree first)
    deg = [(popcount(conflict_mask[e] & base_mask), e) for e in evens_indices]
    deg.sort()
    consider(greedy_complete(greedy_add(base_mask, [e for _, e in deg], DEADLINE), DEADLINE))
    # 4) descending degree
    deg.sort(key=lambda x: -x[0])
    consider(greedy_complete(greedy_add(base_mask, [e for _, e in deg], DEADLINE), DEADLINE))

    # 5) min-degree greedy
    consider(greedy_complete(greedy_min_degree(base_mask, DEADLINE), DEADLINE))
    # 6) max-degree greedy
    consider(greedy_complete(greedy_max_degree(base_mask, DEADLINE), DEADLINE))

    # 7) many random orderings
    for attempt in range(80):
        if time.time() > DEADLINE - 25:
            break
        m = greedy_complete(greedy_random(base_mask, DEADLINE, seed=attempt), DEADLINE)
        consider(m)

    # ---- Local search from best, with several restarts from diverse starts ----
    if time.time() < DEADLINE - 15:
        improved = local_search(best_overall, DEADLINE, max_iters=300000, seed=42)
        consider(improved)

    # A couple of extra local-search restarts from other good starts if time allows
    extra_starts = []
    for attempt in range(8):
        if time.time() > DEADLINE - 10:
            break
        m = greedy_complete(greedy_random(base_mask, DEADLINE, seed=100 + attempt), DEADLINE)
        if popcount(m) >= best_size - 3:
            extra_starts.append(m)

    for idx, m in enumerate(extra_starts):
        if time.time() > DEADLINE - 5:
            break
        improved = local_search(m, DEADLINE, max_iters=50000, seed=200 + idx)
        consider(improved)

    # Final saturate: scan ALL candidates (including odds) for any that still fit
    if time.time() < DEADLINE - 1.0:
        final = best_overall
        for e in range(N):
            if (final >> e) & 1:
                continue
            if (conflict_mask[e] & final) == 0:
                final |= (1 << e)
        if popcount(final) > popcount(best_overall):
            best_overall = final

    result = sorted([i + 1 for i in range(N) if (best_overall >> i) & 1])

    # Final verification; fall back to odds if any conflict slips through
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                return sorted([i + 1 for i in odds_indices])

    return result
# EVOLVE-BLOCK-END
