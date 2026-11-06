"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # 55s wall-clock guard for safety
    
    # Precompute conflicts: for each a, list of (b, c) where a < b <= c <= n and a | (b+c)
    conflicts = {}
    for a in range(1, n + 1):
        conf = []
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if c >= b and c <= n:
                    conf.append((b, c))
        conflicts[a] = conf
    
    # Precompute reverse conflicts for fast delta computation
    # For each x, store:
    #   as_divisor[x]: list of (b,c) in conflicts[x]
    #   as_b[x]: list of (a, c) such that (x, c) in conflicts[a] and a < x
    #   as_c[x]: list of (a, b) such that (b, x) in conflicts[a] and a < x
    as_divisor = conflicts
    as_b = [[] for _ in range(n + 1)]
    as_c = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for (b, c) in conflicts[a]:
            as_b[b].append((a, c))
            as_c[c].append((a, b))
    
    # Fast violation count using bitset representation
    # Use Python int as bitset: bit i set means element (i+1) is in the set
    def set_to_bitset(S):
        bits = 0
        for x in S:
            bits |= (1 << (x - 1))
        return bits
    
    def bitset_to_set(bits):
        S = set()
        for i in range(1, n + 1):
            if bits & (1 << (i - 1)):
                S.add(i)
        return S
    
    def count_violations_bitset(bits):
        """Count violations using bitset."""
        cnt = 0
        for a in range(1, n + 1):
            if not (bits & (1 << (a - 1))):
                continue
            for (b, c) in conflicts[a]:
                if (bits & (1 << (b - 1))) and (bits & (1 << (c - 1))):
                    cnt += 1
        return cnt
    
    # Start with upper tail
    base_set = set(range(171, 257))
    base_bits = set_to_bitset(base_set)
    
    # Greedy add: try adding elements from 1 to 170 in order
    current_set = set(base_set)
    current_bits = base_bits
    
    for x in range(1, 171):
        # Check if adding x causes violations
        new_bits = current_bits | (1 << (x - 1))
        if count_violations_bitset(new_bits) == 0:
            current_set.add(x)
            current_bits = new_bits
    
    best_set = set(current_set)
    best_bits = current_bits
    best_size = len(best_set)
    
    # Local search: try to add more elements by checking all candidates
    def can_add(x, bits):
        """Check if x can be added to set represented by bits without violations."""
        new_bits = bits | (1 << (x - 1))
        # Check violations where x is divisor
        for (b, c) in as_divisor[x]:
            if (bits & (1 << (b - 1))) and (bits & (1 << (c - 1))):
                return False
        # Check violations where x is b
        for (a, c) in as_b[x]:
            if (bits & (1 << (a - 1))) and (bits & (1 << (c - 1))):
                return False
        # Check violations where x is c
        for (a, b) in as_c[x]:
            if (bits & (1 << (a - 1))) and (bits & (1 << (b - 1))):
                return False
        return True
    
    # Repeatedly try to add any element that doesn't cause violations
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, n + 1):
            if not (best_bits & (1 << (x - 1))):
                if can_add(x, best_bits):
                    best_set.add(x)
                    best_bits |= (1 << (x - 1))
                    best_size = len(best_set)
                    changed = True
    
    # Try swaps: remove one element, add another
    if time.time() < deadline:
        for _ in range(2000):  # Limited number of swap attempts
            if time.time() >= deadline:
                break
            if len(best_set) < 1:
                break
            # Pick a random element to remove
            x = random.choice(list(best_set))
            # Try to find a replacement
            for y in range(1, n + 1):
                if y == x or (best_bits & (1 << (y - 1))):
                    continue
                # Remove x, add y
                new_bits = (best_bits & ~(1 << (x - 1))) | (1 << (y - 1))
                if count_violations_bitset(new_bits) == 0:
                    # This swap is valid and maintains same size; try to see if we can add more
                    # Actually, just accept it if it allows adding more later
                    # For now, skip swaps that don't increase size
                    pass
    
    return list(best_set)
# EVOLVE-BLOCK-END
