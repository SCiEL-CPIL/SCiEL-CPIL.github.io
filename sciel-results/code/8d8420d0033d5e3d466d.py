"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 256
    
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    # Precompute prime factors for each number
    prime_factors = {}
    for n in range(1, N + 1):
        pf = set()
        m = n
        d = 2
        while d * d <= m:
            if m % d == 0:
                pf.add(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            pf.add(m)
        prime_factors[n] = pf
    
    # Greedy: start with 1, then add numbers in order of decreasing value
    # A number can be added if its prime factors are disjoint from all selected numbers' prime factors
    
    selected = {1}
    used_primes = set()
    
    # Try adding numbers from largest to smallest
    for n in range(N, 1, -1):
        pf = prime_factors[n]
        if pf.isdisjoint(used_primes):
            selected.add(n)
            used_primes.update(pf)
    
    # Local search: try to improve by swapping
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try to add a number by removing conflicting selected numbers
        for n in range(N, 1, -1):
            if n in selected:
                continue
            pf_n = prime_factors[n]
            # Find selected numbers that share prime factors with n
            conflicting = [s for s in selected if s != 1 and not pf_n.isdisjoint(prime_factors[s])]
            if not conflicting:
                selected.add(n)
                used_primes.update(pf_n)
                improved = True
            else:
                # Try to remove conflicting numbers and add n if it's beneficial
                # Remove the smallest conflicting number
                min_c = min(conflicting)
                if n > min_c:
                    # Check if n's prime factors are disjoint from all non-conflicting selected
                    ok = True
                    for s in selected:
                        if s == 1 or s == min_c:
                            continue
                        if not pf_n.isdisjoint(prime_factors[s]):
                            ok = False
                            break
                    if ok:
                        selected.discard(min_c)
                        selected.add(n)
                        # Update used_primes
                        used_primes.difference_update(prime_factors[min_c])
                        # Re-add primes from n that aren't already used
                        for p in pf_n:
                            if p not in used_primes:
                                used_primes.add(p)
                        # Actually, we need to rebuild used_primes
                        used_primes = set()
                        for s in selected:
                            if s != 1:
                                used_primes.update(prime_factors[s])
                        improved = True
        # Also try to replace a selected number with a larger one that uses the same or subset of primes
        for s in list(selected):
            if s == 1:
                continue
            pf_s = prime_factors[s]
            # Find largest n > s such that prime_factors[n] is a subset of pf_s
            # and n is coprime to all other selected numbers
            for n in range(N, s, -1):
                if n in selected:
                    continue
                pf_n = prime_factors[n]
                if pf_n.issubset(pf_s):
                    # Check coprime to all other selected
                    ok = True
                    for t in selected:
                        if t == 1 or t == s:
                            continue
                        if not pf_n.isdisjoint(prime_factors[t]):
                            ok = False
                            break
                    if ok:
                        selected.discard(s)
                        selected.add(n)
                        used_primes = set()
                        for t in selected:
                            if t != 1:
                                used_primes.update(prime_factors[t])
                        improved = True
                        break
    
    result = sorted(selected)
    if not result:
        result = [1]
    
    return list(result)
# EVOLVE-BLOCK-END
