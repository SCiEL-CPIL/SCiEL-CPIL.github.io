"""
Solver: maximize pairwise-disjoint infinite residue classes with distinct moduli 2..128.
Strategy:
  - Fast exact gcd/criterion checks (precomputed table).
  - Multiple greedy orderings (ascending, descending, prime-first, composite-first,
    bit-count, random shuffles) with a cheap "first valid residue" scan.
  - After each greedy pass, run a fast 1-for-2 local search:
      drop one selected class, try to insert two new moduli.
  - Use a global deadline with safety margin.
  - Return the best (largest) solution found.
"""
import time
import math
import random

MAXM = 128
DEADLINE = 80.0  # safety margin before 90s kill


def _gcd_table():
    g = [[0] * (MAXM + 1) for _ in range(MAXM + 1)]
    for a in range(2, MAXM + 1):
        for b in range(2, MAXM + 1):
            g[a][b] = math.gcd(a, b)
    return g


def _is_disjoint(r1, m1, r2, m2, gt):
    return (r1 - r2) % gt[m1][m2] != 0


def _can_add(r, m, selected, gt):
    for (r2, m2) in selected:
        if not _is_disjoint(r, m, r2, m2, gt):
            return False
    return True


def _greedy(order, gt, time_limit):
    selected = []
    used = set()
    t0 = time.time()
    for m in order:
        if time.time() - t0 > time_limit:
            break
        if m in used:
            continue
        for r in range(m):
            if _can_add(r, m, selected, gt):
                selected.append((r, m))
                used.add(m)
                break
    return selected, used


def _local_search(selected, gt, time_limit):
    """1-for-2 local search: drop one, try to add two."""
    t0 = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - t0 > time_limit:
            break
        n = len(selected)
        for drop_i in range(n):
            if time.time() - t0 > time_limit:
                break
            r_d, m_d = selected[drop_i]
            base = [x for x in selected if x is not selected[drop_i]]
            base_used = set(m2 for (_, m2) in base)
            cand_mods = [m for m in range(2, MAXM + 1) if m not in base_used and m != m_d]
            found = None
            for m1 in cand_mods:
                if found is not None and len(found) == 2:
                    break
                # find first valid residue for m1
                r1_found = None
                for r1 in range(m1):
                    if _can_add(r1, m1, base, gt):
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
                        if _can_add(r2, m2, base, gt):
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


def construct():
    start = time.time()
    deadline = start + DEADLINE
    gt = _gcd_table()

    all_mods = list(range(2, MAXM + 1))
    primes = [m for m in all_mods if _is_prime(m)]
    composites = [m for m in all_mods if not _is_prime(m)]

    orders = []
    orders.append(list(all_mods))
    orders.append(list(reversed(all_mods)))
    orders.append(primes + composites)
    orders.append(composites + primes)
    orders.append(sorted(all_mods, key=lambda m: m.bit_count(), reverse=True))
    orders.append(sorted(all_mods, key=lambda m: m.bit_count()))

    rnd = random.Random(42)
    for _ in range(8):
        o = list(all_mods)
        rnd.shuffle(o)
        orders.append(o)

    best = []
    best_size = 0

    # seed solution: power-of-2 family
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_size = len(seed)

    for order in orders:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        # greedy
        sel, used = _greedy(order, gt, remaining * 0.4)
        if len(sel) > best_size:
            best_size = len(sel)
            best = [list(x) for x in sel]
        # local search
        remaining2 = deadline - time.time()
        if remaining2 > 1.0:
            sel2 = _local_search(sel, gt, remaining2 * 0.6)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best = [list(x) for x in sel2]

    # final polish on best
    if time.time() < deadline and best:
        sel = [tuple(x) for x in best]
        remaining3 = deadline - time.time()
        if remaining3 > 1.0:
            sel2 = _local_search(sel, gt, remaining3 * 0.7)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best = [list(x) for x in sel2]

    if not best:
        best = [[1, 2]]
    return best


if __name__ == "__main__":
    sol = construct()
    print("size:", len(sol))
    print(sol)
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            r1, m1 = sol[i]
            r2, m2 = sol[j]
            g = math.gcd(m1, m2)
            assert (r1 - r2) % g != 0, (i, j, r1, m1, r2, m2)
    mods = [m for (_, m) in sol]
    assert len(set(mods)) == len(mods)
    for r, m in sol:
        assert 0 <= r < m and 2 <= m <= 128
    print("OK")
