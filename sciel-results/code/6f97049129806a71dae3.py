"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    x = int(math.isqrt(n))
    # Verify
    while x * x > n:
        x -= 1
    while (x + 1) * (x + 1) <= n:
        x += 1
    return x

def _square_free_part(n):
    """Return the square-free part of n as a tuple of primes with odd exponent."""
    result = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt % 2 == 1:
                result.append(d)
        d += 1
    if temp > 1:
        result.append(temp)
    return tuple(sorted(result))

def _target_state():
    """The square-free part of 210 = 2*3*5*7."""
    return (2, 3, 5, 7)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    target = _target_state()
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # We want to find a set of numbers whose combined square-free part equals target
    # This is equivalent to: product of square-free parts (XOR in group sense) = target
    # In the group (Z/2Z)^k where k = number of relevant primes, we need the sum to equal target
    
    # Strategy: BFS from target state, applying numbers from 211 upward
    # State: frozenset of primes with odd parity (the "deficit")
    # Start: target state
    # Goal: empty state (all parities even)
    # Each step: pick a number, XOR its square-free part into the current state
    
    # But we also need to track which numbers we've used and minimize the max value
    # Better approach: for each candidate max value M, check if there exists a valid set
    # using only numbers in (210, M] with size <= 100
    
    # Even better: BFS where we track the set of used numbers, but that's too large
    # Instead: for each M from 211 upward, do a BFS/DP to see if we can reach empty state
    
    # The state space is 2^(number of primes up to M). For M up to ~10210, that's too many primes.
    # But we only care about primes that appear in the square-free parts of numbers in (210, M].
    
    # Alternative: Since we just need ANY valid set, and we want to minimize max,
    # let's try small sets first.
    
    # Key observation: If we can find a single number x such that 210*x is a perfect square,
    # then x = 210 (but x must be > 210). So single number won't work directly.
    # 210*x = k^2 => x = k^2/210. For x to be integer, 210 | k^2.
    # 210 = 2*3*5*7, so k must be divisible by 2*3*5*7 = 210.
    # k = 210*m, x = (210*m)^2/210 = 210*m^2.
    # x > 210 => m >= 1 gives x = 210, not > 210. m=2 gives x=840.
    # Check: 210*840 = 176400 = 420^2. Yes!
    # So [840] is a valid solution with max = 840, raw value = 840-210 = 630.
    
    # But can we do better? We need distinct integers > 210.
    # With 2 numbers: 210*a*b = square. a*b must have square-free part = 210's square-free part.
    # So sf(a)*sf(b) (in the group) = (2,3,5,7).
    # We want to minimize max(a,b) where a,b > 210, a != b.
    
    # Let's try to find pairs (a,b) with a < b, both > 210, such that sf(a) XOR sf(b) = target.
    # To minimize b, we want small a and b.
    
    # Let me compute square-free parts for numbers 211, 212, 213, ...
    # and look for pairs whose XOR equals target.
    
    # But the state space for XOR is over primes. Let's encode each square-free part as a frozenset.
    
    # For efficiency, let's precompute square-free parts for numbers 211 to some limit.
    # Then for each M, check if there's a pair (or small set) within (210, M].
    
    # Let's first try pairs:
    best_solution = None
    best_max = max_val + 1
    
    # Precompute square-free parts
    limit = max_val
    sf_parts = {}
    for i in range(211, limit + 1):
        sf_parts[i] = _square_free_part(i)
    
    # For pairs: for each a in 211..M-1, we need b such that sf(b) = sf(a) XOR target
    # XOR of two tuples of primes: symmetric difference
    
    def xor_sf(a, b):
        """Symmetric difference of two sorted tuples of primes."""
        set_a = set(a)
        set_b = set(b)
        result = set_a ^ set_b
        return tuple(sorted(result))
    
    # For each a, compute needed_sf = xor_sf(sf(a), target)
    # Then look for b > a with sf(b) = needed_sf
    
    # Build a map: sf -> list of numbers with that sf
    from collections import defaultdict
    sf_to_nums = defaultdict(list)
    for i in range(211, limit + 1):
        sf_to_nums[sf_parts[i]].append(i)
    
    # Sort each list
    for key in sf_to_nums:
        sf_to_nums[key].sort()
    
    # For pairs
    for a in range(211, limit + 1):
        if time.time() > deadline:
            break
        needed = xor_sf(sf_parts[a], target)
        if needed in sf_to_nums:
            candidates = sf_to_nums[needed]
            # Find smallest b > a
            import bisect
            idx = bisect.bisect_right(candidates, a)
            if idx < len(candidates):
                b = candidates[idx]
                if b <= limit and b != a:
                    if b < best_max:
                        best_max = b
                        best_solution = [a, b]
    
    # Check if single number works (already found 840)
    # 210*m^2 for m=2: 840
    # Verify: 210*840 = 176400, sqrt = 420. Yes.
    if 840 <= limit:
        if 840 < best_max:
            best_max = 840
            best_solution = [840]
    
    # Try triples if needed
    if best_solution is None or best_max > 300:
        # Try to find triples (a,b,c) with a<b<c, all >210, minimizing c
        # sf(a) XOR sf(b) XOR sf(c) = target
        # For each pair (a,b), needed_c_sf = xor_sf(xor_sf(sf(a), sf(b)), target)
        # Look for c > b with that sf
        
        if time.time() < deadline - 5:
            for a in range(211, min(500, limit + 1)):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, min(a + 200, limit + 1)):
                    needed = xor_sf(xor_sf(sf_parts[a], sf_parts[b]), target)
                    if needed in sf_to_nums:
                        candidates = sf_to_nums[needed]
                        import bisect
                        idx = bisect.bisect_right(candidates, b)
                        if idx < len(candidates):
                            c = candidates[idx]
                            if c < best_max and c != a and c != b:
                                best_max = c
                                best_solution = [a, b, c]
    
    # Verify the solution
    if best_solution is not None:
        prod = 210
        for x in best_solution:
            prod *= x
        s = _isqrt(prod)
        if s * s != prod:
            # Fallback to known solution
            best_solution = [840]
            best_max = 840
    
    # Ensure we have a valid solution
    if best_solution is None:
        best_solution = [840]
        best_max = 840
    
    # Verify again
    prod = 210
    for x in best_solution:
        prod *= x
    s = _isqrt(prod)
    if s * s != prod:
        best_solution = [840]
    
    # Check all constraints
    for x in best_solution:
        assert isinstance(x, int)
        assert x > 210
        assert x <= 10210
    assert len(best_solution) >= 1
    assert len(best_solution) <= 100
    assert len(set(best_solution)) == len(best_solution)  # distinct
    
    return list(best_solution)
# EVOLVE-BLOCK-END
