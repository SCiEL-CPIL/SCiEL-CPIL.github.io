"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    # Start with all integers in [0, N], greedily remove least-populated residue
    # class for each prime p <= len(S) in increasing order.
    S = list(range(N + 1))
    k = len(S)
    primes = _primes(N)
    
    for p in primes:
        if p > k:
            break
        # Count residues mod p
        counts = [0] * p
        for v in S:
            counts[v % p] += 1
        # Check if all classes are covered
        if min(counts) > 0:
            # Find the least-populated class
            min_count = min(counts)
            target_class = counts.index(min_count)
            # Remove all elements in that class
            S = [v for v in S if v % p != target_class]
            k = len(S)
    
    return S
# EVOLVE-BLOCK-END
