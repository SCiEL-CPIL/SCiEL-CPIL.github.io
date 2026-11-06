# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    def can_add(x, selected):
        """Check if x can be added to selected without violating constraints."""
        # Check: x does not divide sum of any 3 distinct elements in selected
        for b, c, d in combinations(selected, 3):
            if (b + c + d) % x == 0:
                return False
        
        # Check: for each a in selected, a does not divide x + b + c
        # for any pair b,c in selected with b!=a, c!=a, b!=c
        for a in selected:
            for b, c in combinations(selected, 2):
                if b == a or c == a:
                    continue
                if (x + b + c) % a == 0:
                    return False
        return True
    
    # Try multiple greedy strategies and keep the best
    best = []
    
    # Strategy 1: Descending order (original)
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Ascending order
    selected = []
    for x in range(1, 193):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Only numbers > 96 (large numbers are less likely to divide sums)
    selected = []
    for x in range(192, 96, -1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 4: Numbers in range [65, 192] descending
    selected = []
    for x in range(192, 64, -1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 5: Numbers > 128 descending
    selected = []
    for x in range(192, 128, -1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 6: Alternating large and medium
    selected = []
    candidates = list(range(192, 96, -1)) + list(range(96, 0, -1))
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 7: Primes and large composites
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    selected = []
    primes_large = [x for x in range(192, 96, -1) if is_prime(x)]
    composites_large = [x for x in range(192, 96, -1) if not is_prime(x)]
    for x in primes_large + composites_large:
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    if len(selected) > len(best):
        best = selected[:]
    
    if time.time() > deadline:
        return best
    
    # Strategy 8: Try to fill gaps - start with large, then try to add smaller
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    # Now try to see if we can swap to get more
    if len(selected) > len(best):
        best = selected[:]
    
    return best
# EVOLVE-BLOCK-END
