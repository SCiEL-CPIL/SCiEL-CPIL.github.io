<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _check_c_not_divides_any_product(trial, c):
    """Check c does not divide b*b' for any distinct b,b' in trial."""
    n = len(trial)
    for i in range(n):
        b = trial[i]
        for j in range(i + 1, n):
            if (b * trial[j]) % c == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n_max = P['n']
    
    primes = [a for a in range(2, n_max + 1) if _is_prime(a)]
    composites = [a for a in range(2, n_max + 1) if not _is_prime(a)]
    prime_set = set(primes)
    
    # Pre-compute prime factors for composites
    comp_factors = {}
    for c in composites:
        factors = set()
        tmp = c
        d = 2
        while d * d <= tmp:
            while tmp % d == 0:
                factors.add(d)
                tmp //= d
            d += 1
        if tmp > 1:
            factors.add(tmp)
        comp_factors[c] = factors
    
    best = list(primes)
    best_size = len(best)
    
    def try_remove(removed_primes):
        """Try removing given primes and adding compatible composites."""
        nonlocal best, best_size
        removed_set = set(removed_primes)
        kept = prime_set - removed_set
        trial = [x for x in primes if x not in removed_set]
        trial_size = len(trial)
        
        for c in composites:
            if c in removed_set:
                continue
            cf = comp_factors[c]
            # No kept prime should divide c
            if any(f in kept for f in cf):
                continue
            # Check c doesn't divide any product of two trial elements
            if _check_c_not_divides_any_product(trial, c):
                trial.append(c)
                trial_size += 1
        
        if trial_size > best_size:
            best = trial
            best_size = trial_size
    
    # Single prime removals
    for p in primes:
        if time.time() > deadline:
            break
        try_remove([p])
    
    # Two prime removals
    if time.time() < deadline:
        for p1, p2 in combinations(primes, 2):
            if time.time() > deadline:
                break
            try_remove([p1, p2])
    
    # Three prime removals (sampled)
    if time.time() < deadline:
        import random
        rng = random.Random(42)
        attempts = 0
        while time.time() < deadline and attempts < 50000:
            attempts += 1
            sample = rng.sample(primes, 3)
            try_remove(sample)
    
    result = sorted(best)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _full_check_add(trial, new_elem):
    """Full check: new_elem can be added to trial without violating constraints."""
    n = len(trial)
    # new_elem ∤ b*b' for all distinct b, b' in trial
    for i in range(n):
        b = trial[i]
        for j in range(i + 1, n):
            if (b * trial[j]) % new_elem == 0:
                return False
    # a ∤ new_elem*b for all distinct a, b in trial
    for i in range(n):
        a = trial[i]
        for j in range(n):
            if i == j:
                continue
            b = trial[j]
            if (new_elem * b) % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n_max = P['n']
    
    primes = [a for a in range(2, n_max + 1) if _is_prime(a)]
    composites = [a for a in range(2, n_max + 1) if not _is_prime(a)]
    prime_set = set(primes)
    
    # Pre-compute prime factors for composites
    comp_factors = {}
    for c in composites:
        factors = set()
        tmp = c
        d = 2
        while d * d <= tmp:
            while tmp % d == 0:
                factors.add(d)
                tmp //= d
            d += 1
        if tmp > 1:
            factors.add(tmp)
        comp_factors[c] = factors
    
    best = list(primes)
    best_size = len(best)
    
    def try_remove(removed_primes):
        nonlocal best, best_size
        removed_set = set(removed_primes)
        kept = prime_set - removed_set
        trial = [x for x in primes if x not in removed_set]
        trial_size = len(trial)
        
        # Sort composites by number of prime factors (simpler first)
        for c in composites:
            if c in removed_set:
                continue
            cf = comp_factors[c]
            if any(f in kept for f in cf):
                continue
            if _full_check_add(trial, c):
                trial.append(c)
                trial_size += 1
        
        if trial_size > best_size:
            best = list(trial)
            best_size = trial_size
    
    # Single prime removals
    for p in primes:
        if time.time() > deadline:
            break
        try_remove([p])
    
    # Two prime removals (only pairs where both are small, to limit candidates)
    if time.time() < deadline:
        small_primes = [p for p in primes if p <= 37]
        for p1, p2 in combinations(small_primes, 2):
            if time.time() > deadline:
                break
            try_remove([p1, p2])
    
    # Three prime removals (sampled, focus on small primes)
    if time.time() < deadline:
        import random
        rng = random.Random(42)
        small_primes = [p for p in primes if p <= 23]
        attempts = 0
        while time.time() < deadline and attempts < 30000:
            attempts += 1
            if len(small_primes) >= 3:
                sample = rng.sample(small_primes, 3)
            else:
                sample = rng.sample(primes, 3)
            try_remove(sample)
    
    result = sorted(best)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Hmm, but I'm still concerned about the `_full_check_add` being called many times. Let me think about how many times it's actually called.

For single prime removal (say remove 2): composites with all factors in {2} are: 4, 8, 16, 32, 64, 128. That's 6 calls to `_full_check_add`. Each call is O(53²) ≈ 2800 ops. Total: 6 * 2800 = 16800.

For remove 3: 9, 27, 81. 3 calls.

For remove 2 and 3: composites with factors in {2,3}: 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243. About 24 calls.

The number of pair combinations of small primes (≤37) is C(12,2) = 66. Each with ~24 candidates * 2800 ops = 67K. Total: 66 * 67K = 4.4M. That's fine.

For triple removals of small primes (≤23, 9 primes): C(9,3) = 84 total, but we're sampling 30000 times. Each trial: composites with factors in the 3 removed primes. If we remove {2,3,5}: composites with factors in {2,3,5} up to 256. That's more, maybe 50-80. Each check is O(50²) = 2500. So 80 * 2500 = 200K per trial. 30000 * 200K = 6 billion. Way too much!

I need to limit triple removals much more or make the check faster. Let me reduce to 2000 triple removal attempts and focus on the most promising ones.

Actually, let me reconsider. The most promising removals are those that allow many composites. Removing {2} gives 6 composites. Removing {2,3} gives ~24. Removing {2,3,5} gives more but we lose 3 primes.

Net gain for removing {2}: +6 - 1 = +5
Net gain for removing {2,3}: +24 - 2 = +22 (but many of the 24 might not all be addable due to interactions)
Net gain for removing {2,3,5}: +many - 3

Let me just be conservative with triple removals.

Let me also add a quick pre-check: if c is a power of a single removed prime, it's always addable (as I proved for powers of 2). This saves the full check.

