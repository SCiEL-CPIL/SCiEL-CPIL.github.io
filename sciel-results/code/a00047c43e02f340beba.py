"""Find pairwise-coprime integers from 1..256 maximizing their sum."""
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 75  # 15s safety margin before 90s kill

def construct():
    # Sieve primes up to 256
    n = 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Primes > 128: all can be included, they don't conflict with any composite <= 256
    # because any composite <= 256 has a prime factor <= sqrt(256) = 16
    large_primes = [p for p in primes if p > 128]
    small_primes = [p for p in primes if p <= 128]
    
    # For each composite number <= 256, determine which small primes divide it
    # A composite conflicts with a small prime p if p divides the composite
    # We want to choose a subset of composites and a subset of small primes
    # such that no composite is divisible by any chosen small prime
    # and no two composites share a common factor (but composites are pairwise coprime
    # only if they share no prime factors)
    
    # Actually, the key insight: we need ALL selected numbers to be pairwise coprime.
    # Large primes (>128) are automatically coprime with everything else selected
    # (since any composite <= 256 has prime factors <= 16 < 128, and other large primes
    # are distinct from each other).
    # Small primes: if we include small prime p, we cannot include any multiple of p.
    # Composites: if we include composite c, we cannot include any of its prime factors
    # (which are small primes), and we cannot include any other composite sharing a factor.
    
    # Strategy: 
    # - Always include 1 and all large primes
    # - For small primes and composites, we need to solve a weighted independent set
    #   on a conflict graph where nodes are {small primes} U {composites <= 256}
    #   and edges connect conflicting pairs
    #   But this is complex. Simpler approach:
    #   For each small prime p, we can either:
    #     a) Include p (and exclude all multiples of p)
    #     b) Exclude p, and potentially include some composites divisible by p
    #   But composites have multiple prime factors, making this complex.
    
    # Better approach: Use the fact that composites <= 256 have prime factors <= 16.
    # So only small primes up to 16 matter for conflicts with composites.
    # Primes 17..128 only conflict with their multiples, but their smallest multiple
    # is p^2 or 2p. For p >= 17, 2p >= 34, and p^2 >= 289 > 256, so the only multiples
    # of p in [2,256] are 2p, 3p, ..., but wait, p itself is a prime. The multiples of p
    # that are composite and <= 256 are 2p, 3p, ... but these would share factor p.
    # However, if we include prime p (17 <= p <= 128), we can't include 2p, 3p, etc.
    # But including p gives value p, and including 2p would give value 2p. 
    # Since 2p > p, it might be better to exclude p and include 2p.
    # But 2p might conflict with other numbers.
    
    # Let me reconsider: The optimal solution likely includes:
    # - 1
    # - All large primes > 128
    # - A careful selection of small primes and composites
    
    # Given the complexity, let's use a greedy approach with local search,
    # or use scipy's MILP solver if available.
    
    # Try using scipy.optimize.milp for the small subproblem
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csc_matrix
        import numpy as np
    except ImportError:
        # Fallback to greedy
        return _greedy_solution(large_primes, small_primes, n)
    
    # Build the subproblem: choose from {small primes} U {composites 4..256 that are not prime}
    # such that selected numbers are pairwise coprime
    # This is a maximum weight independent set problem, which is NP-hard in general,
    # but the graph has special structure.
    
    # Actually, let's think differently. For each prime p <= 16, we can decide
    # to either include p, or include some composites divisible by p.
    # But composites can have multiple small prime factors.
    
    # Let's use a different approach: enumerate which small primes to include,
    # and for the rest, find the best composites.
    
    # Small primes up to 16: [2, 3, 5, 7, 11, 13]
    # These are the only primes that can divide composites <= 256
    # (since 17*2 = 34 <= 256, but 17*17 > 256, so composites divisible by 17
    #  would be 17*2=34, 17*3=51, etc., but 34 = 2*17, so it's divisible by both 2 and 17)
    
    # Wait, I need to be more careful. A composite number c <= 256 can have prime factors
    # up to sqrt(c) <= 16, plus possibly one more prime factor. E.g., 2*17 = 34, 3*17 = 51,
    # etc. So primes > 16 can appear as the "large" factor of a composite.
    
    # This makes the problem more complex. Let's just use MILP on the full conflict graph
    # for numbers in {2..256} \ {large primes}, plus the large primes are always included.
    
    # Numbers to consider: 1 (always included), large primes (always included),
    # and the rest: small primes (2..128 that are prime) and composites (4..256 not prime)
    
    # Let me build the candidate set
    candidates = []
    for i in range(2, n + 1):
        if is_prime[i] and i > 128:
            continue  # Always included
        if i == 1:
            continue  # Always included
        candidates.append(i)
    
    m = len(candidates)
    
    # Build conflict matrix: conflict[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    # This is an m x m boolean matrix
    # For MILP: x_i + x_j <= 1 for all conflicting pairs
    # Maximize sum(weight_i * x_i)
    
    # This could be large: m could be up to ~200, giving ~20000 constraints
    # Let's check if it's feasible
    
    # Actually, let's just use a greedy heuristic with local search instead,
    # which should be fast and likely near-optimal.
    
    return _greedy_solution(large_primes, small_primes, n)

def _greedy_solution(large_primes, small_primes, n):
    """Greedy solution with local search."""
    base = [1] + large_primes
    
    # Build conflict graph for remaining numbers
    # Remaining: small primes and composites
    remaining = []
    for i in range(2, n + 1):
        if i in large_primes:
            continue
        remaining.append(i)
    
    # Greedy: pick numbers in order of decreasing value, adding if no conflict
    selected = []
    for x in sorted(remaining, reverse=True):
        conflict = False
        for y in selected:
            if math.gcd(x, y) > 1:
                conflict = True
                break
        if not conflict:
            selected.append(x)
    
    result = base + selected
    
    # Local search: try to swap out one number for a better combination
    # This is a simplified local search
    for _ in range(100):  # Limit iterations
        improved = False
        for i in range(len(result)):
            # Try removing result[i] and adding a better number
            current_val = result[i]
            without_i = result[:i] + result[i+1:]
            best_add = -1
            best_val = -1
            for candidate in range(2, n + 1):
                if candidate in without_i:
                    continue
                if candidate in large_primes:
                    continue  # Already included
                # Check if candidate is coprime with all in without_i
                ok = True
                for y in without_i:
                    if math.gcd(candidate, y) > 1:
                        ok = False
                        break
                if ok and candidate > best_val:
                    best_val = candidate
                    best_add = candidate
            if best_add > 0 and best_val > current_val:
                result = without_i + [best_add]
                improved = True
                break
        if not improved:
            break
    
    return result
