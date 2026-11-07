"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}
_START_TIME = time.time()
_DEADLINE = 55.0  # safety margin before 60s

def _exponent_vector(n, primes):
    """Return exponent vector of n with respect to given primes."""
    vec = []
    for p in primes:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        vec.append(cnt)
    return tuple(vec)

def _subset_products_distinct(vecs):
    """Check if all subset sums of vecs are distinct."""
    seen = {tuple([0]*len(vecs[0]))}
    for i, v in enumerate(vecs):
        new = set()
        for s in seen:
            ns = tuple(s[j] + v[j] for j in range(len(v)))
            new.add(ns)
        seen |= new
        if len(seen) != 2**(i+1):
            return False
    return True

def _subset_products_distinct_fast(vecs):
    """Faster check using integer encoding for small vectors."""
    # Encode each vector as a single integer with base-7 encoding (max exponent is 6 for 2^6=64>40)
    # Actually max exponent for 2 in numbers 2-40: 2^5=32, 2^6=64>40, so max is 5
    # For 3: 3^3=27, 3^4=81>40, so max is 3
    # Let's use base 7 to be safe
    base = 7
    def encode(vec):
        val = 0
        for x in vec:
            val = val * base + x
        return val
    
    seen = {0}
    for v in vecs:
        ev = encode(v)
        new = set()
        for s in seen:
            # We need to add ev to s, but s is encoded... this doesn't work directly
            pass
    # Fall back to tuple approach but with early termination
    seen = {tuple([0]*len(vecs[0]))}
    for i, v in enumerate(vecs):
        new = set()
        for s in seen:
            ns = tuple(s[j] + v[j] for j in range(len(v)))
            new.add(ns)
        if len(new) < 2**i:
            return False
        seen |= new
        if len(seen) != 2**(i+1):
            return False
    return True

def _verify_solution(nums, primes):
    """Verify that all subset products are distinct."""
    vecs = [_exponent_vector(n, primes) for n in nums]
    return _subset_products_distinct(vecs)

def construct():
    t0 = time.time()
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    all_nums = list(range(2, 41))
    
    best = list(primes)  # Start with 12 primes
    best_size = len(best)
    
    # Precompute exponent vectors
    vec_map = {n: _exponent_vector(n, primes) for n in all_nums}
    
    def check_add(current_vecs, new_vec):
        """Check if adding new_vec to current_vecs keeps all subset sums distinct."""
        # Current subset sums
        sums = {tuple([0]*12)}
        for v in current_vecs:
            new_sums = set()
            for s in sums:
                ns = tuple(s[j] + v[j] for j in range(12))
                new_sums.add(ns)
            sums |= new_sums
        # Now add new_vec
        new_sums = set()
        for s in sums:
            ns = tuple(s[j] + new_vec[j] for j in range(12))
            new_sums.add(ns)
        # Check for collision: any sum in new_sums that's already in sums?
        # Actually we need: all 2^(k+1) sums are distinct
        all_sums = sums | new_sums
        expected = 2**(len(current_vecs)+1)
        return len(all_sums) == expected
    
    # Greedy: try to add composites
    composites = [n for n in all_nums if n not in primes]
    
    def greedy_build(order):
        """Build a set by greedily adding numbers in given order."""
        current = list(primes)
        current_vecs = [vec_map[n] for n in primes]
        for n in order:
            if n in current:
                continue
            v = vec_map[n]
            if check_add(current_vecs, v):
                current.append(n)
                current_vecs.append(v)
        return current
    
    # Try multiple random orders
    random.seed(42)
    attempts = 0
    while time.time() - t0 < _DEADLINE - 5:
        attempts += 1
        # Random permutation of composites
        random_comps = composites[:]
        random.shuffle(random_comps)
        result = greedy_build(random_comps)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= P['maximum_size']:
                break
        # Also try some structured orders
        if attempts % 10 == 0:
            # Try adding small composites first
            result2 = greedy_build(sorted(composites))
            if len(result2) > best_size:
                best = result2
                best_size = len(result2)
            # Try adding large composites first
            result3 = greedy_build(sorted(composites, reverse=True))
            if len(result3) > best_size:
                best = result3
                best_size = len(result3)
    
    # Local improvement: try swapping out one element for another
    if time.time() - t0 < _DEADLINE - 3:
        improved = True
        while improved and time.time() - t0 < _DEADLINE - 3:
            improved = False
            current_vecs = [vec_map[n] for n in best]
            for i, n in enumerate(best):
                for m in all_nums:
                    if m in best:
                        continue
                    # Try replacing best[i] with m
                    trial = best[:i] + best[i+1:] + [m]
                    trial_vecs = [vec_map[x] for x in trial]
                    if _subset_products_distinct(trial_vecs):
                        best = trial
                        improved = True
                        break
                if improved:
                    break
    
    # Final verification
    if not _verify_solution(best, primes):
        # Fallback to just primes
        best = list(primes)
    
    return list(best)
# EVOLVE-BLOCK-END
