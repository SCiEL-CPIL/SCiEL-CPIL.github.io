import math
import time

_MAX = 128
_START = time.time()
_DEADLINE = 75.0  # safety margin before 90s kill

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _conflict(r1, m1, r2, m2):
    # classes intersect iff r1 == r2 mod gcd(m1,m2)
    g = _gcd(m1, m2)
    return (r1 % g) == (r2 % g)

def construct():
    t0 = time.time()
    moduli = list(range(2, _MAX + 1))

    # Greedy construction: try to add moduli one by one, picking a residue that
    # is disjoint from all already-chosen classes.
    def try_build(order):
        chosen = []  # list of (r, m)
        for m in order:
            found = False
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    if _conflict(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    found = True
                    break
            if not found:
                pass  # skip this modulus
        return chosen

    # Heuristic ordering: start with small moduli (more restrictive), interleave.
    # Try several orderings and keep the best.
    best = []

    orderings = []
    orderings.append(moduli)                       # ascending
    orderings.append(sorted(moduli, reverse=True)) # descending
    # primes first
    def is_prime(n):
        if n < 2:
            return False
        i = 2
        while i * i <= n:
            if n % i == 0:
                return False
            i += 1
        return True
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    orderings.append(primes + composites)
    orderings.append(composites + primes)
    # ascending by number of divisors (fewer divisors = fewer conflicts)
    def num_divisors(n):
        c = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                c += 1
                if i * i != n:
                    c += 1
            i += 1
        return c
    orderings.append(sorted(moduli, key=lambda m: num_divisors(m)))
    orderings.append(sorted(moduli, key=lambda m: -num_divisors(m)))

    for order in orderings:
        if time.time() - t0 > _DEADLINE:
            break
        cand = try_build(order)
        if len(cand) > len(best):
            best = cand

    # Local improvement: try to swap/insert. Repeatedly attempt to add a skipped
    # modulus by re-choosing a residue, or drop one class to fit two.
    improved = True
    while improved and (time.time() - t0) < _DEADLINE:
        improved = False
        chosen = list(best)
        chosen_moduli = {m for (_, m) in chosen}
        # try adding each missing modulus
        for m in moduli:
            if m in chosen_moduli:
                continue
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    if _conflict(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    improved = True
                    break
            if improved:
                break
        if improved:
            best = chosen

    # Convert to required format [[r, m], ...]
    result = [[r, m] for (r, m) in best]
    # Ensure nonempty fallback
    if not result:
        result = [[1, 2]]
    return result
