"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 256, 'maximum_size': 30}
_DEADLINE = 85.0  # safety margin before 90s kill

def _all_subset_sums(S):
    """Return set of all non-empty subset sums of S."""
    n = len(S)
    sums = set()
    # iterate all non-empty subsets via bitmask
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    return sums

def _no_divisibility_conflict(sums_set):
    """Check that no two distinct values in sums_set divide one another."""
    vals = sorted(sums_set)
    for i in range(len(vals)):
        a = vals[i]
        for j in range(i + 1, len(vals)):
            b = vals[j]
            if b % a == 0:
                return False
    return True

def _is_valid_set(S):
    """Check validity: all subset sums non-empty, no two distinct divide each other,
    and no two different subsets have the same sum (which would cause 2s divisible by s)."""
    if len(S) != len(set(S)):
        return False
    if any(x < 1 or x > 256 for x in S):
        return False
    # Check no two distinct subsets have same sum
    n = len(S)
    seen_sum = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        if s in seen_sum:
            return False  # two different subsets with same sum
        seen_sum[s] = mask
    # Check no two distinct subset sums divide each other
    vals = sorted(seen_sum.keys())
    for i in range(len(vals)):
        a = vals[i]
        for j in range(i + 1, len(vals)):
            b = vals[j]
            if b % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    
    # Strategy: use large primes. Their subset sums are large and close,
    # making it hard for one to divide another (need b >= 2a).
    # Generate primes in range [128, 256]
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in range(128, 257) if is_prime(x)]
    # Sort descending to start with largest
    primes_desc = sorted(primes, reverse=True)
    
    best = [31, 32, 34]  # fallback
    best_size = 3
    
    # Greedy: start with largest primes, add if valid
    # Try multiple starting points
    for start_idx in range(0, min(len(primes_desc), 10)):
        if time.time() - start_time > _DEADLINE:
            break
        current = []
        current_sums = set()
        
        # Start with primes_desc[start_idx]
        for p in primes_desc[start_idx:]:
            if time.time() - start_time > _DEADLINE:
                break
            new_sums = set(current_sums)
            # Add p and all existing sums + p
            new_sums.add(p)
            for s in current_sums:
                new_sums.add(s + p)
            
            # Check: no two distinct values in new_sums divide each other
            valid = True
            vals = sorted(new_sums)
            for i in range(len(vals)):
                a = vals[i]
                for j in range(i + 1, len(vals)):
                    b = vals[j]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.append(p)
                current_sums = new_sums
        
        if len(current) > best_size:
            best = current[:]
            best_size = len(current)
    
    # Also try a few random-ish combinations for variety
    import random
    random.seed(42)
    for attempt in range(50):
        if time.time() - start_time > _DEADLINE:
            break
        # Pick random subset of large primes
        k = random.randint(3, 8)
        candidates = random.sample(primes, min(k, len(primes)))
        candidates.sort(reverse=True)
        
        current = []
        current_sums = set()
        for p in candidates:
            new_sums = set(current_sums)
            new_sums.add(p)
            for s in current_sums:
                new_sums.add(s + p)
            
            valid = True
            vals = sorted(new_sums)
            for i in range(len(vals)):
                a = vals[i]
                for j in range(i + 1, len(vals)):
                    b = vals[j]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.append(p)
                current_sums = new_sums
        
        if len(current) > best_size:
            best = current[:]
            best_size = len(current)
    
    # Final validation
    if _is_valid_set(best):
        return list(best)
    else:
        return [31, 32, 34]
# EVOLVE-BLOCK-END
