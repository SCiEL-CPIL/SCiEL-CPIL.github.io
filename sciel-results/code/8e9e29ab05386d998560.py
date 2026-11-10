"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

# Precompute primes up to 256 and prime factorization info
_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit) if is_prime[i]]

PRIMES = _sieve_primes(_N)

# For each number 1..256, compute its set of prime factors
def _prime_factors(n):
    if n == 1:
        return frozenset()
    factors = set()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return frozenset(factors)

# Precompute prime factors for all numbers
_NUM_FACTORS = [None] + [_prime_factors(i) for i in range(1, _N + 1)]

def construct():
    """
    Find a set of distinct integers from 1..256 such that every pair is coprime,
    maximizing their sum.
    """
    deadline = time.time() + 80
    
    candidates = list(range(2, _N + 1))
    
    # Phase 1: Greedy initialization with multiple orderings
    def greedy_select(order):
        used_primes = set()
        selected = [1]
        total = 1
        for n in order:
            pf = _NUM_FACTORS[n]
            if pf.isdisjoint(used_primes):
                used_primes |= pf
                selected.append(n)
                total += n
        return selected, total, used_primes
    
    orderings = [
        sorted(candidates, key=lambda x: -x),
        sorted(candidates, key=lambda x: (-x / max(len(_NUM_FACTORS[x]), 1), -x)),
        sorted(candidates, key=lambda x: (len(_NUM_FACTORS[x]), -x)),
    ]
    
    best_selected = None
    best_total = 0
    best_used = None
    
    for ordering in orderings:
        sel, tot, used = greedy_select(ordering)
        if tot > best_total:
            best_total = tot
            best_selected = sel
            best_used = used
    
    selected_set = set(best_selected)
    
    # Phase 2: Local search with 1-exchange and 2-exchange
    def get_used_primes(selected):
        used = set()
        for s in selected:
            if s > 1:
                used |= _NUM_FACTORS[s]
        return used
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        all_used = get_used_primes(best_selected)
        
        # 1-exchange: remove a, add b
        for a in list(best_selected):
            if a == 1:
                continue
            freed = _NUM_FACTORS[a]
            remaining_used = all_used - freed
            best_b = None
            best_gain = 0
            for b in candidates:
                if b in selected_set:
                    continue
                pf_b = _NUM_FACTORS[b]
                if pf_b.isdisjoint(remaining_used):
                    gain = b - a
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b is not None and best_gain > 0:
                best_selected.remove(a)
                selected_set.remove(a)
                best_selected.append(best_b)
                selected_set.add(best_b)
                best_total += best_gain
                improved = True
                all_used = get_used_primes(best_selected)
        
        # 2-exchange: remove a1, a2, add b1, b2 (or just b1)
        if time.time() > deadline:
            break
        for i in range(len(best_selected)):
            if best_selected[i] == 1:
                continue
            a1 = best_selected[i]
            for j in range(i + 1, len(best_selected)):
                if best_selected[j] == 1:
                    continue
                a2 = best_selected[j]
                freed = _NUM_FACTORS[a1] | _NUM_FACTORS[a2]
                remaining_used = all_used - freed
                
                # Try adding one number
                best_single = None
                best_single_gain = 0
                for b in candidates:
                    if b in selected_set:
                        continue
                    pf_b = _NUM_FACTORS[b]
                    if pf_b.isdisjoint(remaining_used):
                        gain = b - a1 - a2
                        if gain > best_single_gain:
                            best_single_gain = gain
                            best_single = b
                
                if best_single is not None and best_single_gain > 0:
                    best_selected.remove(a1)
                    best_selected.remove(a2)
                    selected_set.remove(a1)
                    selected_set.remove(a2)
                    best_selected.append(best_single)
                    selected_set.add(best_single)
                    best_total += best_single_gain
                    improved = True
                    all_used = get_used_primes(best_selected)
                    break
            
            if improved and time.time() > deadline:
                break
            if improved:
                break
        
        # Try adding numbers if primes are free
        if time.time() > deadline:
            break
        all_used = get_used_primes(best_selected)
        free_primes = set(PRIMES) - all_used
        if free_primes:
            for n in sorted(candidates, key=lambda x: -x):
                if n in selected_set:
                    continue
                pf = _NUM_FACTORS[n]
                if pf.issubset(free_primes):
                    best_selected.append(n)
                    selected_set.add(n)
                    best_total += n
                    all_used |= pf
                    free_primes -= pf
                    improved = True
    
    # Verify result
    result = sorted(best_selected)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [a for a in range(2, _N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
