import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before the 150 s kill

    # ---- primes up to 1000 ----
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start::step] = [False] * len(is_prime[start::step])
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    n_primes = len(primes)

    # ---- bitset coverage over 1..MAX ----
    MAX = 20000
    FULL = (1 << (MAX + 1)) - 1  # bits 0..MAX

    mask_cache = {}

    def get_mask(p, r):
        key = (p, r)
        m = mask_cache.get(key)
        if m is not None:
            return m
        m = 0
        start_n = p if r == 0 else r
        bit = 1
        # build mask by setting bits at start_n, start_n+p, ...
        for n in range(start_n, MAX + 1, p):
            m |= (1 << n)
        mask_cache[key] = m
        return m

    # precompute all masks for small primes (cheap, avoids repeated big work)
    for p in primes:
        if p <= 300:
            for r in range(p):
                m = 0
                start_n = p if r == 0 else r
                for n in range(start_n, MAX + 1, p):
                    m |= (1 << n)
                mask_cache[(p, r)] = m

    def prefix_of(covered):
        # bit 0 is always set in covered, so first zero bit is >= 1
        unc = (~covered) & FULL
        low = unc & -unc
        if low == 0:
            return MAX
        return low.bit_length() - 2  # first zero at bit k (>=1) -> prefix k-1

    def rebuild_covered(classes):
        cov = 1  # bit 0
        for p, r in classes:
            cov |= get_mask(p, r)
        return cov

    # ---- greedy build: always cover the first gap, choose best prime ----
    def build_greedy(prime_order, time_limit):
        used = set()
        covered = 1
        budget = 1.0
        classes = []
        while time.time() < time_limit:
            m = prefix_of(covered) + 1  # first uncovered integer
            if m > MAX:
                break
            best_key = None
            best = None
            for p in prime_order:
                if p in used:
                    continue
                c = 1.0 / p
                if budget - c < 1e-12:
                    continue
                r = m % p
                nc = covered | get_mask(p, r)
                np = prefix_of(nc)
                newbits = (nc & ~covered).bit_count()
                key = (np, newbits, -c)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (p, r, nc, c)
            if best is None:
                break
            p, r, nc, c = best
            used.add(p)
            classes.append([p, r])
            covered = nc
            budget -= c
        return classes, covered, budget

    # ---- hill climbing: ADD + SWAP ----
    def local_search(classes, budget, time_limit):
        used = {c[0] for c in classes}
        covered = rebuild_covered(classes)
        cur = prefix_of(covered)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # --- ADD ---
            m = cur + 1
            if m <= MAX:
                for p in primes:
                    if p in used:
                        continue
                    c = 1.0 / p
                    if budget - c < 1e-12:
                        continue
                    r = m % p
                    nc = covered | get_mask(p, r)
                    np = prefix_of(nc)
                    if np > cur:
                        classes.append([p, r])
                        used.add(p)
                        budget -= c
                        covered = nc
                        cur = np
                        improved = True
                        break
            if improved:
                continue
            # --- SWAP: remove q, add p ---
            for qi in range(len(classes)):
                q = classes[qi][0]
                cov2 = 1
                for j in range(len(classes)):
                    if j == qi:
                        continue
                    pp, rr = classes[j]
                    cov2 |= get_mask(pp, rr)
                g = prefix_of(cov2) + 1
                if g > MAX:
                    break
                freed = 1.0 / q
                for p in primes:
                    if p in used:
                        continue
                    c = 1.0 / p
                    if budget + freed - c < 1e-12:
                        continue
                    r = g % p
                    nc = cov2 | get_mask(p, r)
                    np = prefix_of(nc)
                    if np > cur:
                        classes[qi] = [p, r]
                        used.remove(q)
                        used.add(p)
                        budget = budget + freed - c
                        covered = nc
                        cur = np
                        improved = True
                        break
                if improved:
                    break
        return classes, budget, cur

    # ---- light residue re-optimisation for the final best solution ----
    def residue_reopt(classes, budget, time_limit):
        used = {c[0] for c in classes}
        covered = rebuild_covered(classes)
        cur = prefix_of(covered)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            covered = rebuild_covered(classes)
            cur = prefix_of(covered)
            m = cur + 1
            # candidate gap positions to try to cover
            gaps = set()
            gaps.add(m)
            for d in range(1, 60, 7):
                gaps.add(m + d)
            for idx in range(len(classes)):
                p, r = classes[idx]
                best_r = r
                best_np = cur
                for g in gaps:
                    cand = g % p
                    if cand == r:
                        continue
                    # recompute coverage with this prime's residue changed
                    cov2 = 1
                    for j in range(len(classes)):
                        pp, rr = classes[j]
                        if j == idx:
                            cov2 |= get_mask(pp, cand)
                        else:
                            cov2 |= get_mask(pp, rr)
                    np = prefix_of(cov2)
                    if np > best_np:
                        best_np = np
                        best_r = cand
                if best_r != r:
                    classes[idx] = [p, best_r]
                    covered = rebuild_covered(classes)
                    cur = best_np
                    improved = True
        return classes, cur

    # ---- exact prefix up to 100000 (for choosing the best / sanity) ----
    def exact_prefix(classes, cap=100000):
        cov = bytearray(cap + 1)
        for p, r in classes:
            start_n = p if r == 0 else r
            for n in range(start_n, cap + 1, p):
                cov[n] = 1
        L = 0
        for i in range(1, cap + 1):
            if cov[i]:
                L = i
            else:
                break
        return L

    # ---- main: restarts + local search, keep best ----
    best_classes = None
    best_score = -1
    best_budget = 0.0
    rng = random.Random(12345)
    attempt = 0
    while time.time() < DEADLINE - 8.0:
        attempt += 1
        order = primes[:]
        rng.shuffle(order)
        tl_build = min(time.time() + 3.0, DEADLINE - 30.0)
        classes, covered, budget = build_greedy(order, tl_build)
        if time.time() > DEADLINE - 25.0:
            break
        tl_ls = min(time.time() + 8.0, DEADLINE - 12.0)
        classes, budget, cur = local_search(classes, budget, tl_ls)
        # score on the 20000-bit prefix (fast proxy)
        if cur > best_score:
            best_score = cur
            best_classes = [c[:] for c in classes]
            best_budget = budget

    if best_classes is None:
        # minimal valid fallback
        best_classes = [[2, 1]]
        best_budget = 0.5

    # final residue re-optimisation on the best solution
    if time.time() < DEADLINE - 4.0:
        best_classes, _ = residue_reopt(best_classes, best_budget, DEADLINE - 2.0)

    # ---- exact budget verification with Fraction; trim if over ----
    total = Fraction(0)
    for p, r in best_classes:
        total += Fraction(1, p)
    if total > Fraction(1, 1):
        # drop the most expensive classes until valid
        best_classes.sort(key=lambda c: -Fraction(1, c[0]))
        for i in range(len(best_classes)):
            if total <= Fraction(1, 1):
                break
            total -= Fraction(1, best_classes[i][0])
            best_classes.pop(i)

    # sanity: ensure distinct primes & valid residues (they are by construction)
    return best_classes
