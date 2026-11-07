"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

DEADLINE = 85.0  # safety margin before 90s kill
_start = time.time()

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}
BASE = P['n']
MAX_N = P['maximum_window']
SIZE = P['maximum_size']

def _deadline_exceeded():
    return (time.time() - _start) > DEADLINE

# Sieve for smallest prime factor
spf = list(range(MAX_N + 1))
for i in range(2, isqrt(MAX_N) + 1):
    if spf[i] == i:
        for j in range(i * i, MAX_N + 1, i):
            if spf[j] == j:
                spf[j] = i

def _sf_set(n):
    """Return frozenset of primes with odd exponent in n."""
    res = set()
    tmp = n
    while tmp > 1:
        p = spf[tmp]
        cnt = 0
        while tmp % p == 0:
            tmp //= p
            cnt += 1
        if cnt % 2 == 1:
            res.add(p)
    return frozenset(res)

# Precompute sf_set for all n in (210, 10210]
sf_cache = {}
for n in range(BASE + 1, MAX_N + 1):
    sf_cache[n] = _sf_set(n)

# Target parity state: sf of 210
target_state = sf_cache[BASE]  # frozenset({2,3,5,7})

def _xor_states(a, b):
    """XOR of two frozensets (symmetric difference)."""
    return a ^ b

def _state_to_sf_val(state):
    """Convert frozenset of primes to integer product."""
    res = 1
    for p in state:
        res *= p
    return res

def construct():
    # Start with 100 smallest integers > 210
    base_set = list(range(BASE + 1, BASE + 1 + SIZE))  # 211..310
    base_set_set = set(base_set)
    
    # Compute current parity state
    cur_state = frozenset()
    for n in base_set:
        cur_state = _xor_states(cur_state, sf_cache[n])
    
    # If already correct, return
    if cur_state == target_state:
        return list(base_set)
    
    # We need to find swaps to fix the state.
    # A single swap: remove x from base, add y not in base.
    # New state = cur_state XOR sf(x) XOR sf(y)
    # We want: cur_state XOR sf(x) XOR sf(y) = target_state
    # => sf(x) XOR sf(y) = cur_state XOR target_state
    delta = _xor_states(cur_state, target_state)
    
    # Index: for each sf value, list of numbers in base with that sf
    sf_to_base = defaultdict(list)
    for n in base_set:
        sf_to_base[sf_cache[n]].append(n)
    
    # Try single swaps: for each y not in base (in increasing order), 
    # check if there exists x in base such that sf(x) = delta XOR sf(y)
    # We want smallest y possible
    for y in range(BASE + 1 + SIZE, MAX_N + 1):  # 311..10210
        if y in base_set_set:
            continue
        needed_sf = _xor_states(delta, sf_cache[y])
        if needed_sf in sf_to_base:
            x = sf_to_base[needed_sf][0]  # pick smallest x with that sf
            new_set = [n for n in base_set if n != x] + [y]
            new_set.sort()
            return new_set
    
    # If no single swap works, try two swaps: remove x1, x2 from base, add y1, y2 not in base
    # New state = cur_state XOR sf(x1) XOR sf(x2) XOR sf(y1) XOR sf(y2) = target_state
    # => sf(x1) XOR sf(x2) XOR sf(y1) XOR sf(y2) = delta
    # This is harder. Let's try: fix y1, y2 (small), then need sf(x1) XOR sf(x2) = delta XOR sf(y1) XOR sf(y2)
    # Precompute all pairs of sf values in base
    base_sfs = [sf_cache[n] for n in base_set]
    
    # Build a map from pair-xor to (x1, x2) for x1 < x2 in base
    pair_xor_to_pair = {}
    for i in range(len(base_set)):
        for j in range(i + 1, len(base_set)):
            px = _xor_states(sf_cache[base_set[i]], sf_cache[base_set[j]])
            if px not in pair_xor_to_pair:
                pair_xor_to_pair[px] = (base_set[i], base_set[j])
    
    # Try pairs of y's in increasing order of max(y1, y2)
    # For efficiency, iterate y1 < y2, and for each, compute needed = delta XOR sf(y1) XOR sf(y2)
    # Check if needed is in pair_xor_to_pair
    # To keep max small, iterate by max value
    for max_y in range(BASE + 1 + SIZE + 1, MAX_N + 1):
        if _deadline_exceeded():
            # Fallback: just return base_set (may be invalid, but better than timeout)
            return list(base_set)
        # y2 = max_y, y1 ranges from BASE+1+SIZE to max_y-1, not in base
        for y1 in range(BASE + 1 + SIZE, max_y):
            if y1 in base_set_set or y1 == max_y:
                continue
            needed = _xor_states(_xor_states(delta, sf_cache[y1]), sf_cache[max_y])
            if needed in pair_xor_to_pair:
                x1, x2 = pair_xor_to_pair[needed]
                new_set = [n for n in base_set if n != x1 and n != x2] + [y1, max_y]
                new_set.sort()
                return new_set
    
    # If we get here, try a more aggressive approach: 
    # Use the fact that we can add "neutral" elements (squares or pairs with same sf)
    # But we already have exactly 100 elements. 
    # Last resort: try random swaps or a broader search
    # For now, return the base set as a fallback (may be invalid)
    return list(base_set)
# EVOLVE-BLOCK-END
