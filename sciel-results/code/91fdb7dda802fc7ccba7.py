"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# ---------- Problem constants ----------
# Primes in 2..40. Every composite in 2..40 is a product of these, so the
# prime-exponent vector (over these primes) fully encodes a number's
# multiplicative identity. Subset-product distinctness <=> the exponent
# vectors form a "sum-free" family (all 2^k subset sums distinct).
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NP = len(PRIMES)
ZERO = tuple(0 for _ in range(NP))

# ---------- Precompute exponent vectors ----------
_VEC = {}
for n in range(2, 41):
    v = [0] * NP
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    _VEC[n] = tuple(v)

# ---------- Verification (exact products, as required) ----------
def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct (exact arbitrary-precision)."""
    k = len(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True


# ---------- Vector-space sum-free check (fast, equivalent) ----------
def _valid_add(sums, vec):
    """True iff adding vec to the current sum-free family keeps it sum-free."""
    for s in sums:
        ns = tuple(s[i] + vec[i] for i in range(NP))
        if ns in sums:
            return False
    return True


def _add(sums, vec):
    """Return the enlarged sum-free family after adding vec (assumes valid)."""
    new = set()
    for s in sums:
        new.add(tuple(s[i] + vec[i] for i in range(NP)))
    return sums | new


# ---------- Known optimal set (16 elements) ----------
KNOWN16 = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def _build_sums(nums):
    sums = {ZERO}
    for n in nums:
        v = _VEC[n]
        new = set()
        for s in sums:
            new.add(tuple(s[i] + v[i] for i in range(NP)))
        sums = sums | new
    return sums


def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    start = time.time()
    deadline = start + 55  # safety margin before the 60/90s wall-clock limit

    # Structural anchor: a verified 16-element solution.
    base = list(KNOWN16)
    if not _verify_subset_products(base):
        # Should never happen, but keep a guaranteed-valid fallback (primes).
        base = [p for p in PRIMES if p <= 40]

    best = base
    best_len = len(base)
    best_sums = _build_sums(best)
    best_set = set(best)

    # ---- Refinement machinery: try to beat 16 with principled searches ----
    # A. Backtracking "add one more" search on top of the base set.
    if time.time() < deadline:
        target = best_len + 1
        found = _bt_add_one(base, target, deadline)
        if found:
            best = found
            best_len = len(found)
            best_sums = _build_sums(best)
            best_set = set(best)

    # B. Multi-restart greedy with varied orderings (different search family).
    if time.time() < deadline:
        found = _greedy_search(deadline)
        if found is not None and len(found) > best_len:
            best = found
            best_len = len(found)
            best_sums = _build_sums(best)
            best_set = set(best)

    # C. Local improvement: repeatedly try to insert any unused number.
    if time.time() < deadline:
        improved = _local_insert(best, best_sums, deadline)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
            best_sums = _build_sums(best)
            best_set = set(best)

    # ---- Final safety: exact verification + type/range/distinct cleanup ----
    if not _verify_subset_products(best):
        best = [p for p in PRIMES if p <= 40]

    result = []
    seen = set()
    for x in best:
        if isinstance(x, bool) or not isinstance(x, int):
            continue
        if not (2 <= x <= 40):
            continue
        if x in seen:
            continue
        seen.add(x)
        result.append(x)

    return result


def _bt_add_one(base, target, deadline):
    """Backtracking search: extend base with (target - len(base)) extra numbers.

    Returns a valid list of size >= target, or None.
    """
    need = target - len(base)
    if need <= 0:
        return list(base)

    sums = _build_sums(base)
    chosen = list(base)
    used = set(base)

    # Candidate pool: all unused numbers, ordered to prefer "independent" ones
    # (numbers with fewer prime factors / rarer primes are safer to add).
    pool = [n for n in range(2, 41) if n not in used]
    pool.sort(key=lambda n: (sum(_VEC[n]), n))

    def bt():
        if time.time() > deadline:
            return False
        if len(chosen) >= target:
            return True
        for idx, n in enumerate(pool):
            if time.time() > deadline:
                return False
            v = _VEC[n]
            if _valid_add(sums, v):
                chosen.append(n)
                new_sums = _add(sums, v)
                # Recurse on the remaining pool (indices after idx to avoid dupes).
                if bt():
                    return True
                chosen.pop()
                pool[idx] = n  # keep pool intact for sibling branches
        return False

    if bt():
        return chosen[:target]
    return None


def _greedy_search(deadline):
    """Multi-restart greedy over different orderings; return best set found."""
    candidates = list(range(2, 41))

    # Several deterministic + randomized orderings (different search families).
    orderings = []

    # primes first (most independent)
    primes_list = [p for p in PRIMES if p <= 40]
    non_primes = [n for n in candidates if n not in primes_list]
    orderings.append(primes_list + non_primes)

    # fewest prime factors first
    orderings.append(sorted(candidates, key=lambda n: (sum(_VEC[n]), n)))

    # primes, then prime powers, then rest
    prime_powers = [4, 9, 16, 25, 27, 32]
    pp = [p for p in prime_powers if 2 <= p <= 40]
    rest = [n for n in candidates if n not in primes_list and n not in pp]
    orderings.append(primes_list + pp + rest)

    # descending size (large primes/composites first)
    orderings.append(sorted(candidates, reverse=True))

    # randomized orderings
    rng = random.Random(12345)
    for _ in range(40):
        o = list(candidates)
        rng.shuffle(o)
        orderings.append(o)

    best = []
    best_len = 0

    def greedy(order):
        sums = {ZERO}
        chosen = []
        for n in order:
            if time.time() > deadline:
                break
            v = _VEC[n]
            if _valid_add(sums, v):
                chosen.append(n)
                sums = _add(sums, v)
        return chosen

    for order in orderings:
        if time.time() > deadline:
            break
        res = greedy(order)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    return best if best else None


def _local_insert(best, best_sums, deadline):
    """Repeatedly try to insert any unused number into the current set."""
    chosen = list(best)
    sums = set(best_sums)
    used = set(chosen)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for n in range(2, 41):
            if n in used:
                continue
            v = _VEC[n]
            if _valid_add(sums, v):
                chosen.append(n)
                sums = _add(sums, v)
                used.add(n)
                improved = True
    return chosen
# EVOLVE-BLOCK-END
