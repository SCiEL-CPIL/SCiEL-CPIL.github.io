"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

# ---------- Precomputation (module-level, done once) ----------
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

N = P['n']
CANDIDATES = list(range(2, N + 1))
PRIMES = _get_primes(N)
NP = len(PRIMES)
PINDEX = {p: i for i, p in enumerate(PRIMES)}

def _factorize(n):
    fac = {}
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1
            n //= p
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac

VECS = {}
for num in CANDIDATES:
    fac = _factorize(num)
    v = [0] * NP
    for p, e in fac.items():
        v[PINDEX[p]] = e
    VECS[num] = v

# Complexity metrics for ordering heuristics
def _total_pf(num):
    n = num
    c = 0
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            c += 1
            n //= p
    if n > 1:
        c += 1
    return c

def _distinct_pf(num):
    n = num
    c = 0
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            c += 1
            while n % p == 0:
                n //= p
    if n > 1:
        c += 1
    return c

TOTAL_PF = {x: _total_pf(x) for x in CANDIDATES}
DISTINCT_PF = {x: _distinct_pf(x) for x in CANDIDATES}

# Deterministic orderings
ORDER_TOTAL_PF = sorted(CANDIDATES, key=lambda x: (TOTAL_PF[x], x))
ORDER_ASC = sorted(CANDIDATES)
ORDER_DESC = sorted(CANDIDATES, reverse=True)
ORDER_PRIMES_FIRST = sorted(CANDIDATES, key=lambda x: (DISTINCT_PF[x], TOTAL_PF[x], x))
ORDER_DPF_DESC = sorted(CANDIDATES, key=lambda x: (-DISTINCT_PF[x], x))

def _valid_set(s):
    """Exact validation: all 2^k subset products distinct."""
    k = len(s)
    prods = {1}
    for x in s:
        new = [p * x for p in prods]
        for np in new:
            if np in prods:
                return False
        prods.update(new)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin within 60s limit

    best_set = []
    best_len = 0

    def greedy(order):
        """Build a valid set greedily using fast vector-based collision checks."""
        current = []
        products = {tuple([0] * NP): 1}  # exponent-vector -> product (for final big-int validation)
        for cand in order:
            v = VECS[cand]
            # Collision iff exists vec in products such that vec + v is also in products
            valid = True
            for vec in products:
                nv = [vec[i] + v[i] for i in range(NP)]
                if tuple(nv) in products:
                    valid = False
                    break
            if valid:
                current.append(cand)
                # add new product vectors
                new_vecs = []
                for vec in products:
                    nv = [vec[i] + v[i] for i in range(NP)]
                    new_vecs.append(tuple(nv))
                products.update(new_vecs)
        return current

    # Precompute a few orderings to reuse
    orderings = [
        ORDER_TOTAL_PF,
        ORDER_ASC,
        ORDER_DESC,
        ORDER_PRIMES_FIRST,
        ORDER_DPF_DESC,
    ]

    # Phase 1: deterministic orderings (fast, no randomness)
    for order in orderings:
        s = greedy(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]

    # Phase 2: random restarts with diverse strategies
    num_restarts = 200000
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        rr = random.random()
        if rr < 0.15:
            order = orderings[0][:]
            random.shuffle(order)
        elif rr < 0.30:
            order = orderings[1][:]
            random.shuffle(order)
        elif rr < 0.45:
            order = orderings[2][:]
            random.shuffle(order)
        elif rr < 0.60:
            order = orderings[3][:]
            random.shuffle(order)
        elif rr < 0.70:
            order = CANDIDATES[:]
            random.shuffle(order)
        elif rr < 0.85:
            # Sort by random key
            keys = {x: random.random() for x in CANDIDATES}
            order = sorted(CANDIDATES, key=lambda x: keys[x])
        else:
            # Sort by total prime factors with random tie-break
            order = sorted(CANDIDATES, key=lambda x: (TOTAL_PF[x], random.random()))

        s = greedy(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]

    # Phase 3: local search — try to add remaining elements to best_set
    if best_set and time.time() < deadline - 3:
        remaining = [c for c in CANDIDATES if c not in best_set]
        improved = best_set[:]
        products = {tuple([0] * NP): 1}
        for x in improved:
            v = VECS[x]
            new_vecs = []
            for vec in products:
                nv = [vec[i] + v[i] for i in range(NP)]
                new_vecs.append(tuple(nv))
            products.update(new_vecs)

        # Try adding any remaining candidate
        added = True
        while added and time.time() < deadline - 3:
            added = False
            for cand in remaining[:]:
                if time.time() > deadline - 3:
                    break
                v = VECS[cand]
                valid = True
                for vec in products:
                    nv = [vec[i] + v[i] for i in range(NP)]
                    if tuple(nv) in products:
                        valid = False
                        break
                if valid:
                    improved.append(cand)
                    new_vecs = []
                    for vec in products:
                        nv = [vec[i] + v[i] for i in range(NP)]
                        new_vecs.append(tuple(nv))
                    products.update(new_vecs)
                    remaining.remove(cand)
                    added = True
                    if len(improved) > best_len:
                        best_len = len(improved)
                        best_set = improved[:]
                    break

        # Try 1-for-1 swaps
        if time.time() < deadline - 2:
            for i in range(len(improved)):
                if time.time() > deadline - 2:
                    break
                removed = improved[i]
                temp_set = improved[:i] + improved[i+1:]
                temp_products = {tuple([0] * NP): 1}
                for x in temp_set:
                    v = VECS[x]
                    new_vecs = []
                    for vec in temp_products:
                        nv = [vec[i2] + v[i2] for i2 in range(NP)]
                        new_vecs.append(tuple(nv))
                    temp_products.update(new_vecs)

                found = False
                for cand in remaining:
                    if time.time() > deadline - 2:
                        break
                    if cand == removed:
                        continue
                    v = VECS[cand]
                    valid = True
                    for vec in temp_products:
                        nv = [vec[i2] + v[i2] for i2 in range(NP)]
                        if tuple(nv) in temp_products:
                            valid = False
                            break
                    if valid:
                        new_set = temp_set + [cand]
                        improved = new_set
                        # rebuild products
                        products = temp_products
                        new_vecs = []
                        for vec in products:
                            nv = [vec[i2] + v[i2] for i2 in range(NP)]
                            new_vecs.append(tuple(nv))
                        products.update(new_vecs)
                        remaining.remove(cand)
                        found = True
                        if len(improved) > best_len:
                            best_len = len(improved)
                            best_set = improved[:]
                        break

                if not found:
                    improved = temp_set + [removed]
                    # rebuild products
                    products = temp_products
                    v = VECS[removed]
                    new_vecs = []
                    for vec in products:
                        nv = [vec[i2] + v[i2] for i2 in range(NP)]
                        new_vecs.append(tuple(nv))
                    products.update(new_vecs)

    # Fallback
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Final exact validation
    if not _valid_set(best_set):
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return list(best_set)
# EVOLVE-BLOCK-END
