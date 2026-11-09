"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import isqrt

N = 210
PRIMES = [2, 3, 5, 7]
TARGET = (1, 1, 1, 1)  # need odd parity for each of 2,3,5,7
MAX_SIZE = 100
DEADLINE_MARGIN = 5.0  # seconds before 90s limit

def _parity_vec(x):
    """Return parity vector for primes [2,3,5,7] of x (other primes must be even - checked separately)."""
    v = [0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        cnt = 0
        while x % p == 0:
            x //= p
            cnt += 1
        v[i] = cnt % 2
    # Check that remaining x is a perfect square (all other primes even)
    if x > 1:
        r = isqrt(x)
        if r * r != x:
            return None  # invalid: has odd-power prime outside our set
    return tuple(v)

def _xor_vec(a, b):
    return tuple((a[i] ^ b[i]) for i in range(4))

def _is_square(x):
    r = isqrt(x)
    return r * r == x

def _verify_solution(nums):
    """Verify: distinct, all > 210, at most 10210, product*210 is perfect square."""
    if not nums:
        return False
    if len(nums) > MAX_SIZE:
        return False
    if len(set(nums)) != len(nums):
        return False
    for x in nums:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > 10210:
            return False
    # Check product * 210 is perfect square via parity
    total_parity = [0, 0, 0, 0]
    for x in nums:
        v = _parity_vec(x)
        if v is None:
            return False
        for i in range(4):
            total_parity[i] ^= v[i]
    # 210 contributes (1,1,1,1)
    for i in range(4):
        total_parity[i] ^= 1
    return all(p == 0 for p in total_parity)

def _search_for_window(K, start_time):
    """Search for a valid solution where all numbers are in (210, 210+K]."""
    candidates = []
    for x in range(211, 211 + K):
        v = _parity_vec(x)
        if v is not None:
            candidates.append((x, v))
    
    if not candidates:
        return None
    
    # Try to find a subset whose XOR equals TARGET
    # Use meet-in-the-middle for small K
    n = len(candidates)
    
    # For very small candidate sets, brute force subsets
    if n <= 20:
        # Try all subsets from smallest size up
        for size in range(1, min(n, MAX_SIZE) + 1):
            for combo in combinations(range(n), size):
                xor = (0, 0, 0, 0)
                for idx in combo:
                    xor = _xor_vec(xor, candidates[idx][1])
                if xor == TARGET:
                    return [candidates[i][0] for i in combo]
        return None
    
    # Meet in the middle
    half = n // 2
    left = candidates[:half]
    right = candidates[half:]
    
    # Build map for right half: parity -> list of indices
    right_map = {}
    for mask in range(1 << len(right)):
        xor = (0, 0, 0, 0)
        for i in range(len(right)):
            if mask & (1 << i):
                xor = _xor_vec(xor, right[i][1])
        if xor not in right_map:
            right_map[xor] = []
        right_map[xor].append(mask)
    
    # Search left half
    for mask in range(1 << len(left)):
        xor = (0, 0, 0, 0)
        for i in range(len(left)):
            if mask & (1 << i):
                xor = _xor_vec(xor, left[i][1])
        need = _xor_vec(xor, TARGET)
        if need in right_map:
            right_mask = right_map[need][0]
            # Build solution
            indices = []
            for i in range(len(left)):
                if mask & (1 << i):
                    indices.append(i)
            for i in range(len(right)):
                if right_mask & (1 << i):
                    indices.append(half + i)
            if indices:
                return [candidates[i][0] for i in indices]
    
    return None

def construct():
    start_time = time.time()
    deadline = 90.0 - DEADLINE_MARGIN
    
    best_solution = None
    best_raw = float('inf')
    
    # Start from the known good solution and try to improve
    # Known: [240, 294] gives raw = 84
    known = [240, 294]
    if _verify_solution(known):
        best_solution = known
        best_raw = max(known) - 210
    
    # Try increasing K from 30 upward
    for K in range(30, 10000):
        if time.time() > deadline:
            break
        sol = _search_for_window(K, start_time)
        if sol is not None:
            raw = max(sol) - 210
            if raw < best_raw:
                best_raw = raw
                best_solution = sol
            # If we found a solution with raw == K, that's optimal for this window
            if raw == K:
                # Can't do better with this window size
                continue
            # Found a solution that doesn't use the full window, 
            # so we might find better with same K? No, we're increasing K.
            # Actually if raw < K, we already found it in a smaller window.
            # So break if we can't improve
            if raw >= K:
                continue
    
    # Also try specific small windows more carefully
    for K in range(1, 90):
        if time.time() > deadline:
            break
        sol = _search_for_window(K, start_time)
        if sol is not None:
            raw = max(sol) - 210
            if raw < best_raw:
                best_raw = raw
                best_solution = sol
    
    if best_solution is None:
        # Fallback: single number 4*210 = 840
        best_solution = [840]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
