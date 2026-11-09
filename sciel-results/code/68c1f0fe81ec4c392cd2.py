import time
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 90s.
    # We want to return well before 90s.
    deadline = start_time + 80.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split
    mid = n // 2
    D1 = denominators[:mid]
    D2 = denominators[mid:]
    
    ones = Fraction(1, 1)
    
    def gen_sums(dens):
        # Returns list of (sum, mask) for subsets of dens with sum < 1
        sums = [(Fraction(0), 0)]
        for i, d in enumerate(dens):
            frac = Fraction(1, d)
            new_sums = []
            for s, m in sums:
                # Not pick
                new_sums.append((s, m))
                # Pick
                ns = s + frac
                if ns < ones:
                    new_sums.append((ns, m | (1 << i)))
            sums = new_sums
            if time.time() > deadline:
                break
        return sums

    sums1 = gen_sums(D1)
    if time.time() > deadline:
        # Fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        if not selected:
            selected = [0]
        return selected

    sums2 = gen_sums(D2)
    if time.time() > deadline:
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        if not selected:
            selected = [0]
        return selected

    # Sort sums2 by value
    sums2.sort(key=lambda x: x[0])
    vals2 = [s[0] for s in sums2]
    
    best_deficit = None
    best_m1 = 0
    best_m2 = 0
    
    for s1, m1 in sums1:
        target = ones - s1
        # Find largest s2 such that s2 < target
        # bisect_left returns index of first element >= target
        idx = bisect.bisect_left(vals2, target)
        if idx > 0:
            s2, m2 = sums2[idx - 1]
            total = s1 + s2
            deficit = ones - total
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_m1 = m1
                best_m2 = m2
        
        if time.time() > deadline:
            break

    if best_deficit is None:
        # Fallback to greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        if not selected:
            selected = [0]
        return selected

    indices = []
    for i in range(len(D1)):
        if best_m1 & (1 << i):
            indices.append(i)
    for i in range(len(D2)):
        if best_m2 & (1 << i):
            indices.append(mid + i)
            
    if not indices:
        indices = [0]
        
    return indices
