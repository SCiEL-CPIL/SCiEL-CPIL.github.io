"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def _is_compatible(a, b, limit=512):
    """Check if lcm(a,b) <= limit."""
    return a * b // math.gcd(a, b) <= limit

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s to be safe within 90s
    
    N = 512
    
    # Build the set of numbers that are "small" in the sense that they divide 512
    # Or more generally, we want a large subset S of [1,512] such that lcm(a,b) <= 512 for all a,b in S.
    
    # Key insight: If we pick all numbers that divide 512, then for any a,b dividing 512,
    # lcm(a,b) also divides 512, so lcm(a,b) <= 512.
    # The divisors of 512 = 2^9 are: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. That's 10 numbers.
    # But we can do better by including numbers that don't divide 512 but still have lcm <= 512 with each other.
    
    # Better approach: Greedy / local search to maximize the size of a compatible set.
    # A set S is compatible if for all a,b in S, lcm(a,b) <= 512.
    
    # Let's think about what numbers can be in such a set.
    # If a > 512, it's not in [1,512], so all elements are <= 512.
    # For lcm(a,b) <= 512, we need a*b/gcd(a,b) <= 512.
    
    # Strategy: Start with a seed set and try to add more elements, or use local search.
    # Alternatively, we can formulate this as a maximum independent set in a graph where
    # two numbers are connected if their lcm > 512, and find a large independent set.
    
    # Given the 60-90s time limit, let's try a greedy approach with some local search.
    
    # First, let's identify which pairs are incompatible.
    # Two numbers a, b are incompatible if lcm(a,b) > 512.
    
    # Let's build a compatibility matrix or at least a way to check compatibility quickly.
    
    # For efficiency, let's precompute for each number which other numbers it's compatible with.
    
    # Actually, let's think differently. The condition lcm(a,b) <= 512 is quite restrictive.
    # Note that lcm(a,b) >= max(a,b), so if we have two numbers both > 256, their lcm is at least 
    # the larger one, which could be up to 512, but lcm(a,b) >= a and >= b.
    # Actually lcm(a,b) >= max(a,b) is not true in general. lcm(6,10) = 30 >= 10 = max(6,10). 
    # But lcm(a,b) is a multiple of both a and b, so lcm(a,b) >= a and lcm(a,b) >= b.
    # Therefore, if a > 512, it can't be in the set. But all are <= 512.
    # If a and b are both close to 512, say 511 and 512, lcm(511,512) = 511*512/gcd(511,512).
    # gcd(511,512) = 1 (consecutive), so lcm = 511*512 which is way > 512.
    
    # So we need to be careful. Let's try a greedy approach:
    # 1. Sort numbers in some order.
    # 2. Try to add each number to the set if it's compatible with all existing members.
    
    # A good heuristic: Start with numbers that have many compatible partners.
    # Or start with the divisors of 512 and try to add more.
    
    # Let's try: Start with all numbers from 1 to 512, and greedily remove incompatible ones.
    # Or better: Start with an empty set and add numbers in order of "compatibility degree".
    
    # Let's compute for each number n, how many other numbers m have lcm(n,m) <= 512.
    # Then sort by this count in descending order and greedily add.
    
    candidates = list(range(1, 513))
    
    # Compute compatibility count for each candidate
    compat_count = {}
    for n in candidates:
        count = 0
        for m in candidates:
            if _is_compatible(n, m, 512):
                count += 1
        compat_count[n] = count
    
    # Sort by compatibility count, descending
    sorted_candidates = sorted(candidates, key=lambda x: (-compat_count[x], x))
    
    # Greedy: add each candidate if compatible with all in current set
    selected = []
    for n in sorted_candidates:
        if all(_is_compatible(n, s, 512) for s in selected):
            selected.append(n)
        if time.time() > deadline:
            break
    
    # Now try local search: swap elements to improve
    # Or try to add more by removing some and adding others
    
    # Let's also try a different greedy: start with small numbers
    selected2 = []
    for n in range(1, 513):
        if all(_is_compatible(n, s, 512) for s in selected2):
            selected2.append(n)
        if time.time() > deadline:
            break
    
    # Take the better of the two
    if len(selected2) > len(selected):
        selected = selected2
    
    # Try local search: for each element in selected, try to swap with a non-selected element
    # to see if we can get a larger set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        selected_set = set(selected)
        non_selected = [n for n in candidates if n not in selected_set]
        
        for i, a in enumerate(list(selected)):
            for b in non_selected:
                if _is_compatible(a, b, 512):
                    # Try swapping a with b
                    new_selected = [x for x in selected if x != a]
                    new_selected.append(b)
                    if all(_is_compatible(x, y, 512) for x, y in combinations(new_selected, 2)):
                        selected = new_selected
                        improved = True
                        break
            if improved:
                break
        if time.time() > deadline:
            break
    
    # Ensure nonempty
    if not selected:
        selected = [1]
    
    return selected
# EVOLVE-BLOCK-END
