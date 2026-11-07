"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_START = time.time()
_DEADLINE = _START + 85.0  # evaluator kills at 90s; 5s safety margin

# ---------- Prime Exponent Structure ----------
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PN = len(_PRIMES)
_EXP_LIST = {}   # n -> list of exponents over _PRIMES

def _build_exp_table():
    for n in range(2, 41):
        m = n
        vec = [0] * _PN
        for i, p in enumerate(_PRIMES):
            while m % p == 0:
                vec[i] += 1
                m //= p
        _EXP_LIST[n] = vec

_build_exp_table()

# ---------- Conflict Graph (rational dependence of exponent vectors) ----------
# If two exponent vectors are scalar multiples, the two numbers are powers of a
# common base and can NEVER coexist in a valid set. Pre-compute so we can reject
# such pairs instantly without any product arithmetic.

def _are_rationally_dependent(a, b):
    va = _EXP_LIST[a]
    vb = _EXP_LIST[b]
    ia = -1
    for i in range(_PN):
        if va[i] != 0:
            ia = i
            break
    ib = -1
    for i in range(_PN):
        if vb[i] != 0:
            ib = i
            break
    if ia == -1 or ib == -1:
        return False
    if ia != ib:
        return False
    pivot_a = va[ia]
    pivot_b = vb[ib]
    for i in range(_PN):
        if va[i] * pivot_b != vb[i] * pivot_a:
            return False
    return True

_CONFLICT = {n: set() for n in range(2, 41)}
for _a in range(2, 41):
    for _b in range(_a + 1, 41):
        if _are_rationally_dependent(_a, _b):
            _CONFLICT[_a].add(_b)
            _CONFLICT[_b].add(_a)

# ---------- Core Validity Machinery ----------

def _can_add(prods, x):
    """True iff multiplying the existing product set by x adds only new products.
    Collision exists iff some q in prods satisfies q/x in prods (i.e. q = (q/x)*x).
    Avoids materializing p*x for every product."""
    for q in prods:
        if q % x == 0 and (q // x) in prods:
            return False
    return True

def _extend(prods, x):
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return new_prods

def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        prods = _extend(prods, x)
    return prods

def _is_valid_set(lst):
    return len(_subset_products_set(lst)) == (1 << len(lst))

# ---------- Construction ----------

def _greedy_construct(order):
    selected = []
    prods = {1}
    used = set()
    for x in order:
        if len(selected) >= 18:
            break
        if x in used:
            continue
        # instant structural rejection
        if any(s in _CONFLICT[x] for s in selected):
            continue
        if _can_add(prods, x):
            selected.append(x)
            used.add(x)
            prods = _extend(prods, x)
    return selected, prods

def _random_initial(rng):
    order = list(range(2, 41))
    rng.shuffle(order)
    return _greedy_construct(order)

# ---------- Local Search ----------

def _local_search(selected, prods, rng):
    """Try to grow / improve the set via add, 1-for-1 swap, and 2-for-1 swap moves."""
    selected = list(selected)
    prods = _subset_products_set(selected)
    size = len(selected)
    for _round in range(60):
        if time.time() > _DEADLINE or size >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]

        # 1) pure add
        for x in candidates:
            if time.time() > _DEADLINE or size >= 18:
                break
            if any(s in _CONFLICT[x] for s in selected):
                continue
            if _can_add(prods, x):
                selected.append(x)
                used.add(x)
                prods = _extend(prods, x)
                size += 1
                improved = True
                candidates = [c for c in range(2, 41) if c not in used]

        if size >= 18:
            break

        # 2) 1-for-1 swap: drop r, add x (net zero size but may unlock growth)
        for r in list(selected):
            if time.time() > _DEADLINE:
                break
            base = _subset_products_set([s for s in selected if s != r])
            base_size = len(base)
            used2 = used - {r}
            for x in range(2, 41):
                if x in used2:
                    continue
                if any(s in _CONFLICT[x] for s in selected if s != r):
                    continue
                if _can_add(base, x):
                    new_sel = [s for s in selected if s != r] + [x]
                    # only accept if it lets us add at least one more afterwards
                    prods2 = _extend(base, x)
                    grew = False
                    for y in range(2, 41):
                        if y in new_sel:
                            continue
                        if any(s in _CONFLICT[y] for s in new_sel):
                            continue
                        if _can_add(prods2, y):
                            selected = new_sel + [y]
                            prods = _extend(prods2, y)
                            used = set(selected)
                            size = len(selected)
                            grew = True
                            improved = True
                            break
                    if grew:
                        break

        # 3) 2-for-1 swap: drop two, add one (escape plateaus)
        if not improved and size >= 2:
            for i in range(size):
                if time.time() > _DEADLINE:
                    break
                for j in range(i + 1, size):
                    if time.time() > _DEADLINE:
                        break
                    base = _subset_products_set([s for k, s in enumerate(selected) if k != i and k != j])
                    used3 = used - {selected[i], selected[j]}
                    for x in range(2, 41):
                        if x in used3:
                            continue
                        if any(s in _CONFLICT[x] for s in selected if s not in (selected[i], selected[j])):
                            continue
                        if _can_add(base, x):
                            new_sel = [s for k, s in enumerate(selected) if k != i and k != j] + [x]
                            prods2 = _extend(base, x)
                            # accept if it enables a subsequent add (net +1)
                            for y in range(2, 41):
                                if y in new_sel:
                                    continue
                                if any(s in _CONFLICT[y] for s in new_sel):
                                    continue
                                if _can_add(prods2, y):
                                    selected = new_sel + [y]
                                    prods = _extend(prods2, y)
                                    used = set(selected)
                                    size = len(selected)
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return selected, _subset_products_set(selected)

def construct():
    rng = random.Random()

    # Start from a known-valid 16-set seed, then try to push toward 18.
    seed = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    if not _is_valid_set(seed):
        seed = [p for p in range(2, 41) if not _CONFLICT[p] or True][:12]
        seed = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    best = seed[:]
    best_size = len(seed)

    # Polish the seed with local search
    polished, _ = _local_search(seed, None, rng)
    if len(polished) > best_size:
        best, best_size = polished[:], len(polished)

    # Random restarts
    while time.time() < _DEADLINE:
        sel, _ = _random_initial(rng)
        if len(sel) > best_size:
            sel, _ = _local_search(sel, None, rng)
        if len(sel) > best_size:
            best, best_size = sel[:], len(sel)
            if best_size >= 18:
                break
        # Occasionally polish the current best too
        if time.time() < _DEADLINE - 5:
            p, _ = _local_search(best, None, rng)
            if len(p) > best_size:
                best, best_size = p[:], len(p)
                if best_size >= 18:
                    break

    # Final safety verification; fall back to a guaranteed-valid set if needed.
    if not _is_valid_set(best):
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
