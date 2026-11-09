"""Optimized search for finite sets U maximizing sumset/difference-set exponent bound."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _set_sum_diff(U):
    """Return (|U+U|, |U-U|) for a finite set U of ints."""
    Us = sorted(U)
    n = len(Us)
    plus = set()
    minus = set()
    for i in range(n):
        a = Us[i]
        for j in range(i, n):
            plus.add(a + Us[j])
        for j in range(n):
            minus.add(a - Us[j])
    return len(plus), len(minus)

def _score(U):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    nplus, nminus = _set_sum_diff(U)
    if nplus == 0:
        return 0.0, sorted(U)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = random.Random(42)

    best_val = -1e9
    best_set = [0, 1, 3]

    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _score(U)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_set = s

    # Exhaustive search of small sets (finds U={0,1,2,4} with score ~1.028)
    for n in range(2, 8):
        for combo in combinations(range(1, 20), n - 1):
            U = [0] + list(combo)
            consider(U)

    # Structured families
    # Powers of 2 (superincreasing)
    for n in range(2, 20):
        U = [2**k for k in range(n)]
        consider(U)

    # Powers of 3
    for n in range(2, 18):
        U = [3**k for k in range(n)]
        consider(U)

    # Powers of b for various bases
    for base in [2, 3, 4, 5, 7, 10]:
        for n in range(2, 15):
            U = [base**k for k in range(n)]
            consider(U)

    # Triangular numbers
    for n in range(2, 25):
        U = [k*(k+1)//2 for k in range(n)]
        consider(U)

    # Quadratic
    for n in range(2, 25):
        U = [k*k for k in range(n)]
        consider(U)

    # Fibonacci
    fibs = [0, 1]
    for k in range(2, 30):
        fibs.append(fibs[-1] + fibs[-2])
    for n in range(2, 20):
        consider(fibs[:n])

    # Near-lacunary: lacunary with small perturbations
    for base in [2, 3]:
        for n in range(3, 12):
            base_set = [base**k for k in range(n)]
            # Try adding small offsets
            for offset in [1, 2, 3]:
                U = [x + offset if k > 0 else 0 for k, x in enumerate(base_set)]
                if len(set(U)) == len(U):
                    consider(U)

    # Random sets
    for _ in range(3000):
        if time.time() > deadline:
            break
        n = rng.randint(3, 15)
        maxv = rng.choice([10, 20, 50, 100, 200, 500, 1000])
        vals = rng.sample(range(1, maxv + 1), min(n - 1, maxv))
        consider([0] + vals)

    # Local search
    cur = list(best_set)
    improved = True
    it = 0
    while improved and time.time() < deadline - 2.0 and it < 800:
        improved = False
        it += 1
        mx = max(cur) if cur else 0
        # Generate candidates
        cands = set()
        for c in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]:
            cands.add(c)
        cands.add(mx + 1)
        cands.add(mx * 2 + 1)
        cands.add(mx * 3 + 1)
        cands.add(mx + mx // 2)
        cands.add(max(1, mx // 2))
        cands.add(max(1, mx // 3))
        cands.add(mx + 2)
        cands.add(mx * 2)
        cands.add(mx * 2 - 1)
        cands.add(mx + 3)
        cands.add(mx * 3 - 1)
        cands.add(mx * 2 + 2)
        # Random candidates
        for _ in range(30):
            cands.add(rng.randint(1, max(2, mx * 4)))
        cands = [c for c in cands if c >= 0 and c not in cur]
        rng.shuffle(cands)
        
        for c in cands:
            U = cur + [c]
            v, s = _score(U)
            if v > best_val + 1e-12:
                best_val = v
                best_set = s
                cur = s
                improved = True
                break
        
        # Try removals
        if not improved:
            for c in list(cur):
                if c == 0:
                    continue
                U = [x for x in cur if x != c]
                v, s = _score(U)
                if v > best_val + 1e-12:
                    best_val = v
                    best_set = s
                    cur = s
                    improved = True
                    break

        # Try swaps
        if not improved and len(cur) > 2:
            for i in range(1, len(cur)):
                for c in [cur[i]-1, cur[i]+1, cur[i]*2, cur[i]+2, cur[i]-2]:
                    if c <= 0 or c in cur:
                        continue
                    U = list(cur)
                    U[i] = c
                    v, s = _score(U)
                    if v > best_val + 1e-12:
                        best_val = v
                        best_set = s
                        cur = s
                        improved = True
                        break
                if improved:
                    break

    # Final random perturbation search
    if time.time() < deadline - 1.0:
        for _ in range(3000):
            if time.time() > deadline - 0.5:
                break
            # Perturb best set
            cur = list(best_set)
            op = rng.choice(['add', 'remove', 'swap', 'add_remove'])
            if op == 'add':
                c = rng.randint(1, max(2, max(cur) * 3))
                if c not in cur:
                    cur.append(c)
            elif op == 'remove' and len(cur) > 2:
                idx = rng.randint(1, len(cur) - 1)
                cur.pop(idx)
            elif op == 'swap' and len(cur) > 2:
                idx = rng.randint(1, len(cur) - 1)
                cur[idx] = rng.randint(1, max(2, max(cur) * 3))
                cur = list(set(cur))
                if 0 not in cur:
                    cur.append(0)
            elif op == 'add_remove' and len(cur) > 3:
                idx = rng.randint(1, len(cur) - 1)
                cur.pop(idx)
                c = rng.randint(1, max(2, max(cur) * 3))
                if c not in cur:
                    cur.append(c)
            consider(cur)

    return best_set
# EVOLVE-BLOCK-END
