"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _squarefree_kernel_bits(x):
    """Return bitmask of primes {2,3,5,7} that divide x an odd number of times.
    Also return True if x has no other prime factors (i.e., kernel is subset of {2,3,5,7})."""
    # Check if x is of form 2^a * 3^b * 5^c * 7^d * k^2 for some integer k
    # i.e., the square-free part of x only involves primes 2,3,5,7
    n = x
    bits = 0
    # Count exponent of 2
    cnt = 0
    while n % 2 == 0:
        n //= 2
        cnt += 1
    if cnt % 2 == 1:
        bits |= 1
    # Count exponent of 3
    cnt = 0
    while n % 3 == 0:
        n //= 3
        cnt += 1
    if cnt % 2 == 1:
        bits |= 2
    # Count exponent of 5
    cnt = 0
    while n % 5 == 0:
        n //= 5
        cnt += 1
    if cnt % 2 == 1:
        bits |= 4
    # Count exponent of 7
    cnt = 0
    while n % 7 == 0:
        n //= 7
        cnt += 1
    if cnt % 2 == 1:
        bits |= 8
    # Now n should be a perfect square if x's square-free kernel is subset of {2,3,5,7}
    if n == 1:
        return bits, True
    # Check if n is a perfect square
    r = math.isqrt(n)
    if r * r == n:
        return bits, True
    return bits, False

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _find_subset_xor(candidates, target):
    """Given list of (x, bits), find a subset whose XOR of bits equals target.
    Returns list of x values, or None."""
    # Use Gaussian elimination to find a basis, then express target in terms of basis
    # We want a subset of size <= 100
    # Simple approach: use linear algebra to find any solution
    
    # Build augmented system: for each candidate, we have a vector (bits) and want to select subset
    # XOR of selected = target
    
    # Use standard GF(2) Gaussian elimination to find a solution
    n_vars = len(candidates)
    if n_vars == 0:
        return None
    
    # We'll do elimination on the bit vectors (4 bits)
    # Actually, since we only have 4 bits, the space is at most 16-dimensional
    # We can just find which target is achievable and extract a small subset
    
    # Better: find a basis of the column space, then express target
    # Each candidate has a 4-bit vector. We want subset XOR = target.
    
    # Simple greedy: try to build up the target
    # Use the fact that we have at most 16 distinct bit patterns
    
    # Collect all candidates by their bit pattern
    from collections import defaultdict
    by_bits = defaultdict(list)
    for x, bits in candidates:
        by_bits[bits].append(x)
    
    # We need to find a multiset of bit patterns whose XOR is target
    # Since XOR is its own inverse, we can use each pattern at most once effectively
    # (using two of same pattern cancels out)
    
    # Find a subset of the available bit patterns that XOR to target
    # Available patterns are the keys of by_bits
    available_patterns = list(by_bits.keys())
    
    # Try all subsets of available_patterns (at most 16 patterns, so 2^16 = 65536)
    # But we want a small subset. Let's try greedy first.
    
    # Greedy: repeatedly pick a pattern that reduces the remaining target
    # Actually, let's just do BFS on the state space of 4-bit values
    # State: current XOR value (0-15), we want to reach target
    # From state s, we can go to s XOR p for any available pattern p
    
    # BFS to find shortest path from 0 to target
    from collections import deque
    # dist[state] = list of patterns used to reach state
    dist = {0: []}
    queue = deque([0])
    while queue:
        cur = queue.popleft()
        if cur == target:
            # Extract the actual numbers
            patterns_used = dist[cur]
            result = []
            for p in patterns_used:
                # Pick the smallest x with this pattern
                result.append(by_bits[p][0])
            return result
        for p in available_patterns:
            nxt = cur ^ p
            if nxt not in dist:
                dist[nxt] = dist[cur] + [p]
                queue.append(nxt)
    
    return None

def construct():
    deadline = time.time() + 55  # safety margin
    
    target = 15  # kernel of 210 = 2*3*5*7
    
    # Binary search on M = max selected
    lo = 211
    hi = 10210
    
    best_result = None
    best_M = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        # Collect all x in (210, mid] with valid kernel
        candidates = []
        for x in range(211, mid + 1):
            bits, valid = _squarefree_kernel_bits(x)
            if valid:
                candidates.append((x, bits))
        
        # Check if we can find a subset with XOR = target and size <= 100
        subset = _find_subset_xor(candidates, target)
        if subset is not None and len(subset) <= P['maximum_size']:
            # Verify
            prod = 1
            for x in subset:
                prod *= x
            prod *= P['n']
            if _is_perfect_square(prod):
                best_result = subset
                best_M = mid
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    if best_result is not None:
        # Ensure all are Python int
        return [int(x) for x in best_result]
    
    # Fallback: return the seed answer
    return [int(4 * P['n'])]
# EVOLVE-BLOCK-END
