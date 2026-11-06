"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _doubly_covered_prefix(residues, primes, cap=100000):
    """Largest L such that every x in 1..L satisfies >=2 congruences.
    Returns L (capped at cap)."""
    counts = [0] * (cap + 1)
    # For each prime p with residue r, mark x in [1..cap] where x % p == r
    for r, p in zip(residues, primes):
        # first x >= 1 with x % p == r
        # x = r + k*p, need x >= 1
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        # iterate
        x = start
        while x <= cap:
            counts[x] += 1
            x += p
    L = 0
    for x in range(1, cap + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues, primes):
    return _doubly_covered_prefix(residues, primes)

def construct():
    global _DEADLINE
    primes = _primes_up_to(1000)
    n = len(primes)
    assert n == 168, f"expected 168 primes, got {n}"

    # Seed: the i-th prime covers ceil(i/2)
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best = residues[:]
    best_score = _score(best, primes)

    # Local search: try swapping residue of one prime with another value to improve prefix
    # We do a simple hill-climb: for each prime index, try changing its residue to a value
    # that helps cover the first gap.
    def first_gap(res):
        L = _doubly_covered_prefix(res, primes, cap=best_score + 200)
        return L

    # Try to extend: find the first integer not doubly covered, then try to fix it
    # by adjusting residues of primes that pass through that integer.
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        L = _doubly_covered_prefix(best, primes, cap=min(100000, best_score + 5000))
        if L >= 100000:
            break
        # Find first gap
        gap = L + 1
        if gap > 100000:
            break
        # Find primes p where gap % p == r_p (i.e., primes that already cover gap)
        # and primes where we can change r_p to gap % p to cover gap.
        # We want to add coverage to gap without breaking coverage elsewhere.
        # Try: for each prime index i, set residues[i] = gap % primes[i], check if score improves.
        # Only try a limited number of candidates per iteration to save time.
        tried = 0
        for i in range(n):
            if time.time() > _DEADLINE - 5:
                break
            p = primes[i]
            new_r = gap % p
            if new_r == best[i]:
                continue
            old = best[i]
            best[i] = new_r
            # Quick check: does this help? Compute prefix up to gap+some
            L2 = _doubly_covered_prefix(best, primes, cap=min(100000, gap + 500))
            if L2 > best_score:
                best_score = L2
                improved = True
                # continue searching from new state
                break
            else:
                best[i] = old
            tried += 1
            if tried > 200:
                break
    return best
# EVOLVE-BLOCK-END
