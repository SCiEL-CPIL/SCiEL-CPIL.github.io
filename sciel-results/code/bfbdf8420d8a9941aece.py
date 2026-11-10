import time
import math
import random

def construct():
    start_time = time.time()
    # Explicit deadline guard with safety margin before the 90s wall-clock kill
    deadline = start_time + 80.0
    MAX_MOD = 128

    # Precompute gcd table for moduli 2..128
    gt = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for a in range(2, MAX_MOD + 1):
        for b in range(2, MAX_MOD + 1):
            gt[a][b] = math.gcd(a, b)

    all_mods = list(range(2, MAX_MOD + 1))

    def _is_disjoint(r1, m1, r2, m2):
        return (r1 - r2) % gt[m1][m2] != 0

    def _can_add(r, m, selected):
        for (r2, m2) in selected:
            if not _is_disjoint(r, m, r2, m2):
                return False
        return True

    def _greedy(order, time_limit):
        selected = []
        used = set()
        t0 = time.time()
        for m in order:
            if time.time() - t0 > time_limit:
                break
            if m in used:
                continue
            for r in range(m):
                if _can_add(r, m, selected):
                    selected.append((r, m))
                    used.add(m)
                    break
        return selected

    def _local_search(selected, time_limit):
        """1-for-2 local search: drop one, try to insert two new moduli."""
        selected = list(selected)
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            n = len(selected)
            for drop_i in range(n):
                if time.time() - t0 > time_limit:
                    break
                r_d, m_d = selected[drop_i]
                base = [x for i, x in enumerate(selected) if i != drop_i]
                base_used = set(m2 for (_, m2) in base)
                cand_mods = [m for m in range(2, MAX_MOD + 1) if m not in base_used]
                found = None
                for m1 in cand_mods:
                    if found is not None and len(found) == 2:
                        break
                    r1_found = None
                    for r1 in range(m1):
                        if _can_add(r1, m1, base):
                            r1_found = r1
                            break
                    if r1_found is None:
                        continue
                    base.append((r1_found, m1))
                    base_used.add(m1)
                    for m2 in cand_mods:
                        if m2 == m1 or m2 in base_used:
                            continue
                        r2_found = None
                        for r2 in range(m2):
                            if _can_add(r2, m2, base):
                                r2_found = r2
                                break
                        if r2_found is not None:
                            found = [(r1_found, m1), (r2_found, m2)]
                            break
                    base.pop()
                    base_used.discard(m1)
                    if found is not None:
                        break
                if found is not None:
                    new_sel = base + found
                    if len(new_sel) > n:
                        selected = new_sel
                        improved = True
                        break
        return selected

    def _is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [m for m in all_mods if _is_prime(m)]
    composites = [m for m in all_mods if not _is_prime(m)]

    # Build diverse orderings
    orders = []
    orders.append(list(all_mods))
    orders.append(list(reversed(all_mods)))
    orders.append(primes + composites)
    orders.append(composites + primes)
    orders.append(sorted(all_mods, key=lambda m: m.bit_count(), reverse=True))
    orders.append(sorted(all_mods, key=lambda m: m.bit_count()))

    rnd = random.Random(42)
    for _ in range(12):
        o = list(all_mods)
        rnd.shuffle(o)
        orders.append(o)

    # Seed with a known valid solution (powers of 2 family)
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_size = len(seed)

    for order in orders:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1.0:
            break

        # Greedy construction
        sel = _greedy(order, remaining * 0.4)
        if len(sel) > best_size:
            best_size = len(sel)
            best = [list(x) for x in sel]

        # Local search refinement
        remaining2 = deadline - time.time()
        if remaining2 > 1.0:
            sel2 = _local_search(sel, remaining2 * 0.6)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best = [list(x) for x in sel2]

    # Final polish on the best solution found
    if time.time() < deadline and best:
        sel = [tuple(x) for x in best]
        remaining3 = deadline - time.time()
        if remaining3 > 1.0:
            sel2 = _local_search(sel, remaining3 * 0.7)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best = [list(x) for x in sel2]

    if not best:
        best = [[1, 2]]
    return best
