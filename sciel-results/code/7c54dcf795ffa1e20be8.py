"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import itertools
import random

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i, v in enumerate(is_prime) if v]

def _is_smooth(n, primes_set):
    """Check if n is P-smooth where P is the set of primes in primes_set."""
    temp = n
    for p in primes_set:
        if p * p > temp:
            break
        while temp % p == 0:
            temp //= p
    if temp > 1:
        # temp is a prime factor not in primes_set (or a power of one)
        if temp not in primes_set:
            return False
    return True

def construct():
    """Find 12 distinct positive integers <= 10000 minimizing distinct primes in pairwise sums."""
    
    all_primes = _sieve_primes(20000)
    
    # Try prime sets in increasing order of size
    # For each prime set, use greedy clique finding
    for num_primes in range(1, 10):
        # Try the first num_primes primes (smallest primes are most likely to work)
        primes_tuple = tuple(all_primes[:num_primes])
        primes_set = set(primes_tuple)
        
        # Precompute which sums from 2 to 20000 are P-smooth
        smooth_sums = [False] * 20001
        for s in range(2, 20001):
            smooth_sums[s] = _is_smooth(s, primes_set)
        
        # Greedy clique: try multiple starting strategies
        # Strategy 1: try small numbers first
        # Strategy 2: try numbers that are themselves P-smooth
        # Strategy 3: random restarts
        
        best_result = None
        
        # Try greedy with small numbers
        selected = []
        for a in range(1, 10001):
            if len(selected) >= 12:
                break
            compatible = True
            for b in selected:
                s = a + b
                if s > 20000 or not smooth_sums[s]:
                    compatible = False
                    break
            if compatible:
                selected.append(a)
        
        if len(selected) >= 12:
            best_result = selected[:12]
        
        # Try greedy with P-smooth numbers first
        smooth_numbers = [a for a in range(1, 10001) if _is_smooth(a, primes_set)]
        selected = []
        for a in smooth_numbers:
            if len(selected) >= 12:
                break
            compatible = True
            for b in selected:
                s = a + b
                if s > 20000 or not smooth_sums[s]:
                    compatible = False
                    break
            if compatible:
                selected.append(a)
        
        if len(selected) >= 12 and (best_result is None):
            best_result = selected[:12]
        
        # Try a few random restarts
        for _ in range(20):
            random_start = random.randint(1, 10000)
            candidates = list(range(random_start, min(random_start + 2000, 10001)))
            candidates += list(range(1, random_start))
            random.shuffle(candidates)
            
            selected = []
            for a in candidates:
                if len(selected) >= 12:
                    break
                compatible = True
                for b in selected:
                    s = a + b
                    if s > 20000 or not smooth_sums[s]:
                        compatible = False
                        break
                if compatible:
                    selected.append(a)
            
            if len(selected) >= 12:
                if best_result is None:
                    best_result = selected[:12]
        
        if best_result is not None:
            # Verify
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    s = best_result[i] + best_result[j]
                    if s > 20000 or not smooth_sums[s]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return best_result
        
        if len(selected) >= 12:
            result = selected[:12]
            # Verify all pairwise sums
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    s = result[i] + result[j]
                    if s > 20000 or not smooth_sums[s]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result
    
    # Fallback: use even numbers but verify the actual prime count
    # Actually, let's try a smarter fallback: all multiples of 2
    # Sums of two even numbers are even, but may have other prime factors
    # Let's just return 12 small even numbers and let the evaluator measure
    result = list(range(2, 26, 2))
    return result

# EVOLVE-BLOCK-END
