import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # safety margin under 150s

    N = 512
    # weights[a] = 1/(512-a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Prime factors of each a in 1..511
    def prime_factors(n):
        fs = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                fs.add(d)
                n //= d
            d += 1
        if n > 1:
            fs.add(n)
        return fs

    factors = [set() for _ in range(N)]
    for a in range(1, N):
        factors[a] = prime_factors(a)

    # conflict[a] = bitmask of b (1..511) with gcd(a,b)>1
    conflict = [0] * N
    for a in range(1, N):
        for b in range(a + 1, N):
            if factors[a] & factors[b]:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)

    def score_mask(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            s += weights[a]
            m ^= lsb
        return s

    def get_chosen(mask):
        chosen = []
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            chosen.append(a)
            m ^= lsb
        return chosen

    # Greedy seed with given order
    def greedy(order):
        mask = 0
        for a in order:
            if (conflict[a] & mask) == 0:
                mask |= (1 << a)
        return mask

    # Local search: best-improvement over moves, time-budgeted
    def local_search(mask, deadline):
        best = mask
        best_score = score_mask(mask)
        
        # Precompute lists for speed
        all_nodes = list(range(1, N))
        
        while time.time() < deadline:
            improved = False
            chosen = get_chosen(mask)
            chosen_set = set(chosen)
            unchosen = [a for a in all_nodes if a not in chosen_set]

            # ADD move: add best x compatible
            add_gain = 0.0
            add_x = -1
            for x in unchosen:
                if (conflict[x] & mask) == 0:
                    if weights[x] > add_gain:
                        add_gain = weights[x]
                        add_x = x
            if add_x > 0:
                mask |= (1 << add_x)
                best = mask
                best_score += add_gain
                improved = True
                continue

            # 1-for-1 swap
            swap_gain = 0.0
            swap_a = -1
            swap_x = -1
            for a in chosen:
                base = mask ^ (1 << a)
                for x in unchosen:
                    if (conflict[x] & base) == 0:
                        g = weights[x] - weights[a]
                        if g > swap_gain:
                            swap_gain = g
                            swap_a = a
                            swap_x = x
            if swap_x > 0:
                mask = (mask ^ (1 << swap_a)) | (1 << swap_x)
                best = mask
                best_score += swap_gain
                improved = True
                continue

            # 1-for-2: remove a, add x,y
            p12_gain = 0.0
            p12_a = -1
            p12_x = -1
            p12_y = -1
            for a in chosen:
                base = mask ^ (1 << a)
                cands = [x for x in unchosen if (conflict[x] & base) == 0]
                if len(cands) < 2:
                    continue
                cands.sort(key=lambda t: weights[t], reverse=True)
                cands = cands[:50]
                for i in range(len(cands)):
                    x = cands[i]
                    for j in range(i + 1, len(cands)):
                        y = cands[j]
                        if (conflict[x] & (1 << y)) == 0:
                            g = weights[x] + weights[y] - weights[a]
                            if g > p12_gain:
                                p12_gain = g
                                p12_a = a
                                p12_x = x
                                p12_y = y
            if p12_x > 0:
                mask = (mask ^ (1 << p12_a)) | (1 << p12_x) | (1 << p12_y)
                best = mask
                best_score += p12_gain
                improved = True
                continue

            # 2-for-1: remove a,b, add x
            p21_gain = 0.0
            p21_a = -1
            p21_b = -1
            p21_x = -1
            nc = len(chosen)
            for i in range(nc):
                a = chosen[i]
                for j in range(i + 1, nc):
                    b = chosen[j]
                    base = mask ^ (1 << a) ^ (1 << b)
                    for x in unchosen:
                        if (conflict[x] & base) == 0:
                            g = weights[x] - weights[a] - weights[b]
                            if g > p21_gain:
                                p21_gain = g
                                p21_a = a
                                p21_b = b
                                p21_x = x
            if p21_x > 0:
                mask = (mask ^ (1 << p21_a)) ^ (1 << p21_b) | (1 << p21_x)
                best = mask
                best_score += p21_gain
                improved = True
                continue

            # 2-for-2: remove a,b, add x,y
            p22_gain = 0.0
            p22_a = -1
            p22_b = -1
            p22_x = -1
            p22_y = -1
            for i in range(nc):
                a = chosen[i]
                for j in range(i + 1, nc):
                    b = chosen[j]
                    base = mask ^ (1 << a) ^ (1 << b)
                    cands = [x for x in unchosen if (conflict[x] & base) == 0]
                    if len(cands) < 2:
                        continue
                    cands.sort(key=lambda t: weights[t], reverse=True)
                    cands = cands[:30]
                    for k in range(len(cands)):
                        x = cands[k]
                        for l in range(k + 1, len(cands)):
                            y = cands[l]
                            if (conflict[x] & (1 << y)) == 0:
                                g = weights[x] + weights[y] - weights[a] - weights[b]
                                if g > p22_gain:
                                    p22_gain = g
                                    p22_a = a
                                    p22_b = b
                                    p22_x = x
                                    p22_y = y
            if p22_x > 0:
                mask = (mask ^ (1 << p22_a)) ^ (1 << p22_b) | (1 << p22_x) | (1 << p22_y)
                best = mask
                best_score += p22_gain
                improved = True
                continue

            if not improved:
                break

        return best, best_score

    # Multiple diverse greedy seeds
    best_mask = None
    best_score = -1.0

    # Strategy 1: descending weight (prefer high a)
    desc_order = list(range(N - 1, 0, -1))
    mask = greedy(desc_order)
    s = score_mask(mask)
    if s > best_score:
        best_score = s
        best_mask = mask

    # Strategy 2: ascending weight (prefer low a, but coprime)
    asc_order = list(range(1, N))
    mask = greedy(asc_order)
    s = score_mask(mask)
    if s > best_score:
        best_score = s
        best_mask = mask

    # Strategy 3: primes first, then composites by weight
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [a for a in range(1, N) if is_prime(a)]
    composites = [a for a in range(1, N) if not is_prime(a)]
    primes.sort(key=lambda a: -weights[a])
    composites.sort(key=lambda a: -weights[a])
    prime_first = primes + composites
    mask = greedy(prime_first)
    s = score_mask(mask)
    if s > best_score:
        best_score = s
        best_mask = mask

    # Strategy 4: random weighted seeds
    rng = random.Random(42)
    for trial in range(5):
        if time.time() > DEADLINE - 20.0:
            break
        order = list(range(1, N))
        # Assign random priorities with weight bias
        weighted = []
        for a in range(1, N):
            w = weights[a] * (0.5 + rng.random())
            weighted.append((w, a))
        weighted.sort(key=lambda x: -x[0])
        order = [a for _, a in weighted]
        mask = greedy(order)
        s = score_mask(mask)
        if s > best_score:
            best_score = s
            best_mask = mask

    # Local search on best seed
    deadline_inner = DEADLINE - 10.0
    best_mask, best_score = local_search(best_mask, deadline_inner)

    # Try more random seeds with shorter local search
    rng2 = random.Random(9999)
    for s_idx in range(8):
        if time.time() > deadline_inner - 2.0:
            break
        order = list(range(1, N))
        weighted = []
        for a in range(1, N):
            w = weights[a] * (0.3 + 0.7 * rng2.random())
            weighted.append((w, a))
        weighted.sort(key=lambda x: -x[0])
        order = [a for _, a in weighted]
        mask = greedy(order)
        seed_deadline = min(time.time() + 4.0, deadline_inner - 2.0)
        mask, score = local_search(mask, seed_deadline)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Final polish
    if time.time() < DEADLINE - 2.0:
        best_mask, best_score = local_search(best_mask, DEADLINE - 1.0)

    result = get_chosen(best_mask)
    return result
