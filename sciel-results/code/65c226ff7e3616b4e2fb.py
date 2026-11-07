"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

DEADLINE_SECONDS = 80.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS

    N = 256

    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]

    # "Safe" primes: p > N//2, i.e., p > 128.
    # For any distinct a,b,c in S, if p is one of them and a,b are the other two,
    # then a*b <= 256 * 255 = 65280. But if p > 128, can p divide a*b?
    # We need a*p | b*c or b*p | a*c or c*p | a*b for the TRIPLE (a,b,c) to be invalid.
    # Actually, the constraint is: for every triple (a,b,c), a does NOT divide b*c.
    # If p is a member and p > 128, consider triple (p, x, y) where x,y are other members.
    # We need: p does NOT divide x*y.
    # Since x,y <= 256, x*y <= 65536. But p > 128.
    # If p is prime > 128, then p divides x*y iff p divides x or p divides y.
    # But x,y <= 256 < 2p, so p divides x iff x == p (impossible, distinct) or x=0.
    # So p does NOT divide x*y for any distinct x,y < 2p.
    # Thus ALL primes > 128 are safe to include.
    safe_primes = [p for p in primes if p > 128]

    # The remaining candidates: all numbers 2..256 that are NOT safe primes.
    # We need to pick a subset of these such that:
    # 1. For every triple (a,b,c) in the subset, a does not divide b*c.
    # 2. For every triple (p, x, y) where p is a safe prime and x,y are from our subset,
    #    p does not divide x*y (automatically satisfied as shown above).
    # 3. For every triple (a, p, x) where a is from subset, p is safe prime, x is from subset:
    #    a does not divide p*x. Since p is prime and p > 128 >= a (a <= 128 or a is composite <=256),
    #    wait, a can be up to 256. Let me reconsider.
    #
    # Actually, the constraint applies to ALL triples. Let me be more careful.
    # For a triple (a, b, c) with a,b,c distinct members:
    #   - a does not divide b*c
    #   - b does not divide a*c
    #   - c does not divide a*b
    #
    # If p is a safe prime (>128) and a, x are other members:
    #   Triple (p, a, x): p does not divide a*x. As shown, this is true.
    #   Triple (a, p, x): a does not divide p*x. Since gcd(a,p) might share factors...
    #     a divides p*x iff a/gcd(a,p) divides x. Since p is prime, gcd(a,p) is 1 or p.
    #     If p does not divide a (a < p or a not multiple of p, and a <= 256 < 2p so a not multiple of p unless a=p),
    #     then gcd(a,p)=1, so a divides p*x iff a divides x.
    #     If a divides x and a,x are in our subset, that's a problem for the triple (a,p,x).
    #   Similarly for triple (x, p, a): x does not divide p*a, so x does not divide a (if gcd(x,p)=1).
    #
    # So the safe primes introduce additional constraints: for any a,x in our subset,
    # a must not divide x AND x must not divide a (when both < p for all safe primes p).
    # Actually this is stronger than just the triple constraint among subset members.
    #
    # Wait, let me reconsider. The constraint among subset members already requires that
    # for triple (a, b, c) in subset, a does not divide b*c. This is a WEAKER condition than
    # a does not divide b. So the safe primes impose a STRONGER pairwise non-divisibility.
    #
    # Hmm, this makes the problem harder. Let me just brute force search over the non-safe numbers.

    # Let me take a simpler approach: greedy with local improvement.
    # Start with all primes > 128 (they're safe among themselves and with small numbers).
    # Then greedily add other numbers if they don't violate the constraint.

    base = set(safe_primes)

    def check_add(s, candidate):
        """Check if adding candidate to set s violates the constraint."""
        for a in s:
            for b in s:
                if a == b:
                    continue
                # Triple (candidate, a, b): candidate does not divide a*b
                if a * b % candidate == 0:
                    return False
                # Triple (a, candidate, b): a does not divide candidate*b
                if candidate * b % a == 0:
                    return False
                # Triple (a, b, candidate): b does not divide a*candidate
                if a * candidate % b == 0:
                    return False
        return True

    # Greedy: try to add numbers from 2 to 256
    candidates = [x for x in range(2, N + 1) if x not in base]
    # Sort by "difficulty" - prefer numbers with fewer conflicts
    # For now, just try in increasing order, then try descending
    for order in [range(2, N + 1), range(N, 1, -1)]:
        if time.time() > deadline - 2:
            break
        current = set(base)
        for x in order:
            if x in current:
                continue
            if check_add(current, x):
                current.add(x)

        if len(current) > len(base):
            base = current

    # Local improvement: try to swap or add more
    best = list(base)

    # Try single additions
    for x in range(2, N + 1):
        if x in base:
            continue
        if check_add(base, x):
            base.add(x)
            best = list(base)

    # Try to remove one element and add two
    if time.time() < deadline - 5:
        best_size = len(base)
        best_set = set(base)
        for x in list(base):
            if time.time() > deadline - 3:
                break
            trial = base - {x}
            added = 0
            for y in range(2, N + 1):
                if y in trial or y == x:
                    continue
                if check_add(trial, y):
                    trial.add(y)
                    added += 1
                    if len(trial) > best_size:
                        best_size = len(trial)
                        best_set = set(trial)
            if added >= 2 and len(trial) > best_size:
                best_size = len(trial)
                best_set = set(trial)

        base = best_set

    result = sorted(base)
    # Ensure types are Python int and result is a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
