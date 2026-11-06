"""Optimized odd covering system search using bitwise maximum coverage."""

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    base = BASE
    divisors = _get_divisors(base)
    
    # Precompute masks for each (m, r) pair
    # mask is a Python integer with bits set for residues in the class
    # bit i is 1 if i % m == r
    candidates = []
    for m in divisors:
        for r in range(m):
            mask = 0
            # Set bits for r, r+m, r+2m, ...
            # Since m divides base, the number of elements is base // m
            # We can construct the mask efficiently
            # But constructing it bit by bit is slow.
            # Alternative: use a loop to set bits.
            # Given base=51975, max m is 51975.
            # Max bits is 51975.
            # Number of candidates is sum(m for m in divisors) ~ 100k.
            # Constructing masks might take a bit of time but is one-time cost.
            
            # Efficient mask construction:
            # We can use a string of '1' and '0' or just loop.
            # Looping over base/m elements for each of ~100k candidates
            # Total operations ~ sum(base/m * m) = sum(base) = 47 * 51975 ~ 2.4M ops.
            # This is fast in Python.
            
            for i in range(r, base, m):
                mask |= (1 << i)
            candidates.append((m, r, mask))
    
    # Sort candidates by potential coverage (number of bits) descending
    # This is base // m
    candidates.sort(key=lambda x: -base // x[0])
    
    # Initial uncovered mask: all bits 0 to base-1 set
    # (1 << base) - 1
    uncovered = (1 << base) - 1
    
    used_m = set()
    classes = []
    
    # Greedy loop
    # We have 47 moduli. We pick at most 47 classes.
    for _ in range(len(divisors)):
        best_gain = -1
        best_idx = -1
        
        for i, (m, r, mask) in enumerate(candidates):
            if m in used_m:
                continue
            # Intersection of mask and uncovered
            intersection = mask & uncovered
            gain = intersection.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain <= 0:
            break
        
        m, r, mask = candidates[best_idx]
        used_m.add(m)
        classes.append([m, r])
        
        # Update uncovered
        uncovered &= ~mask
    
    return classes
