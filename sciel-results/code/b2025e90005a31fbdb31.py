# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85.0  # 85s safety margin (limit is 90s)
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute lcm for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    def has_bad_triple(selected, x):
        """Check if adding x to selected creates a bad triple."""
        n = len(selected)
        for i in range(n):
            a = selected[i]
            lax = get_lcm(a, x)
            for j in range(i + 1, n):
                b = selected[j]
                if get_lcm(a, b) == lax and get_lcm(b, x) == lax:
                    return True
        return False
    
    def greedy(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if not has_bad_triple(selected, x):
                selected.append(x)
        return selected
    
    # Try multiple orderings
    best = []
    
    # Ordering 1: ascending
    s1 = greedy(nums)
    if len(s1) > len(best):
        best = s1
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 2: descending
    s2 = greedy(nums[::-1])
    if len(s2) > len(best):
        best = s2
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 3: odd first, then even
    s3 = greedy(sorted(nums, key=lambda x: (x & 1, x)))
    if len(s3) > len(best):
        best = s3
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 4: by number of prime factors (more factors = more constraints, add later)
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    s4 = greedy(sorted(nums, key=omega))
    if len(s4) > len(best):
        best = s4
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 5: by value descending among odds, then evens
    s5 = greedy(sorted(nums, key=lambda x: (x & 1, -x)))
    if len(s5) > len(best):
        best = s5
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 6: powers of 2 first (they rarely form bad triples with each other)
    def is_pow2(x):
        return (x & (x - 1)) == 0
    s6 = greedy(sorted(nums, key=lambda x: (not is_pow2(x), x)))
    if len(s6) > len(best):
        best = s6
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Local search improvement: try adding remaining elements
    improved = list(best)
    remaining = set(nums) - set(improved)
    
    # Try adding each remaining element one at a time
    for x in sorted(remaining, key=lambda v: omega(v)):
        if time.time() > deadline:
            break
        if not has_bad_triple(improved, x):
            improved.append(x)
            remaining.discard(x)
    
    if len(improved) > len(best):
        best = improved
    
    # Swap-based local search: try removing one element and adding two that conflict
    # with only the removed element
    improved2 = list(best)
    remaining2 = set(nums) - set(improved2)
    
    # Build conflict graph: for each pair in improved2, check if they conflict with x
    # For each element e in improved2, find which remaining elements conflict with e
    # Try: remove e, add x and y where x,y only conflict with e (not with each other or others)
    
    if time.time() < deadline - 5:
        # More sophisticated local search
        # Try all single removals and see if we can add more than one element
        best_size = len(improved2)
        for e in list(improved2):
            if time.time() > deadline:
                break
            candidate = [x for x in improved2 if x != e]
            # Try to greedily add remaining elements that don't conflict with candidate
            added = []
            for x in sorted(remaining2, key=lambda v: omega(v)):
                if not has_bad_triple(candidate + added, x):
                    added.append(x)
            if len(candidate) + len(added) > best_size:
                improved2 = candidate + added
                best_size = len(improved2)
                remaining2 = set(nums) - set(improved2)
    
    if len(improved2) > len(best):
        best = improved2
    
    return best if best else [1]
# EVOLVE-BLOCK-END
