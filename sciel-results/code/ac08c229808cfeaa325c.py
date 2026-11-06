"""Rewritten program: greedy conflict-graph / class-removal maximization for admissible tuples in [0, 5000]."""

N = 5000


def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    # Primes up to N (we only ever need p <= len(tuple) <= N+1, so N suffices).
    primes = _primes(N)

    # Start from a dense seed interval. The known asymptotic for the largest
    # admissible tuple of length ~ n has width ~ n * log n. Use a width that
    # is generous so the repair step can shed a moderate number of points.
    width = int(N * 1.2) + 100  # ~6100, capped to stay within [0, N]
    width = min(width, N + 1)
    points = list(range(0, width))

    # Repair: for every prime p <= len(points), ensure at least one residue
    # class mod p is missed. Process primes in an order that minimizes the
    # number of removals: larger p first (they have more classes, so the
    # least-populated class is smaller on average), and within that, remove
    # the least-populated class.
    # We iterate to a fixed point because removing points can change len(points)
    # and thus the set of relevant primes.
    changed = True
    while changed:
        changed = False
        n = len(points)
        # Only primes p <= n matter.
        relevant = [p for p in primes if p <= n]
        # Order: larger primes first (more classes -> cheaper to break),
        # tie-break by current "damage" (size of least-populated class).
        # Build class counts once per prime is expensive, so do a cheap
        # pre-pass ordering by p descending, then refine.
        relevant.sort(reverse=True)
        for p in relevant:
            # Count residues.
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            # If all classes are covered, remove the least-populated one.
            if min(counts) > 0:
                # Find a residue class with minimal count.
                target = min(range(p), key=lambda r: counts[r])
                # Remove all points in that class.
                points = [v for v in points if v % p != target]
                changed = True
                # len(points) shrank; break and re-evaluate relevant primes.
                break

    # Final safety: guarantee admissibility for every prime p <= len(points).
    n = len(points)
    for p in [q for q in primes if q <= n]:
        present = set(v % p for v in points)
        if len(present) == p:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            target = min(range(p), key=lambda r: counts[r])
            points = [v for v in points if v % p != target]

    return points
