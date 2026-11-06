"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = [False] * len(range(start, limit + 1, step))
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 330s wall-clock limit

    points = list(range(N + 1))
    n = len(points)

    # Primes up to N+1 (tuple length won't exceed N+1)
    all_primes = _primes(N + 1)

    # Greedy: repeatedly find a prime where all residue classes are covered,
    # and remove the least-populated class.
    while True:
        if time.time() > deadline:
            break

        n = len(points)
        if n < 2:
            break

        # Only need primes up to n
        relevant_primes = [p for p in all_primes if p <= n]

        violation_found = False
        for p in relevant_primes:
            # Count occurrences of each residue class
            counts = [0] * p
            for v in points:
                counts[v % p] += 1

            if 0 in counts:
                # At least one class is empty, admissible for this p
                continue

            # All classes filled: find the least-populated class
            min_count = min(counts)
            if min_count == 0:
                continue
            # Remove points in the least-populated class(es)
            # Pick the class with minimum count; tie-break by lowest residue
            target_class = counts.index(min_count)
            points = [v for v in points if v % p != target_class]
            violation_found = True
            break

        if not violation_found:
            break

    # Final check: ensure admissibility
    # (The loop should have ensured this, but verify)
    n = len(points)
    for p in [p for p in all_primes if p <= n]:
        residues = set(v % p for v in points)
        if len(residues) == p:
            # Still a violation - shouldn't happen, but repair
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            target_class = counts.index(min(counts))
            points = [v for v in points if v % p != target_class]
            break

    return points

# EVOLVE-BLOCK-END
