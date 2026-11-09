"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s wall-clock limit
    
    N = 255
    target = 255
    target_primes = set()
    t = target
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if t % p == 0:
            target_primes.add(p)
            while t % p == 0:
                t //= p
    
    # All candidate numbers must share a factor with 255 (i.e., divisible by 3, 5, or 17)
    # and must be pairwise non-coprime with each other.
    
    # Strategy: Start with multiples of 3 (85 elements), then try to improve by
    # removing some small multiples of 3 and adding numbers divisible by 5 or 17.
    
    # Build the set of all numbers in [1,255] that share a factor with 255
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, target) > 1:
            candidates.append(x)
    
    # Precompute compatibility: two numbers are compatible if gcd > 1
    # We want the largest clique in the graph where edges connect compatible pairs,
    # and the clique must include 255.
    
    # Since 255 must be in the set, every other element must share a factor with 255.
    # So all elements must be divisible by 3, 5, or 17.
    
    # Key insight: Let's partition candidates by their "type" based on which of {3,5,17} divide them.
    # Type 3: divisible by 3 only (not 5, not 17)
    # Type 5: divisible by 5 only (not 3, not 17)  
    # Type 17: divisible by 17 only (not 3, not 5)
    # Type 15: divisible by 3 and 5 (not 17)
    # Type 51: divisible by 3 and 17 (not 5)
    # Type 85: divisible by 5 and 17 (not 3)
    # Type 255: divisible by 3, 5, and 17
    
    # If we take all multiples of 3 (types 3, 15, 51, 255), that's 85 elements.
    # Can we do better?
    
    # If we include a type-5 number (div by 5, not 3), it must share a factor with ALL other elements.
    # It shares factor 5 with type-5, type-15, type-85, type-255 numbers.
    # It does NOT share a factor with type-3 or type-17 numbers (unless they happen to share another factor).
    # So if we include a type-5 number, we can't include type-3 or type-17 numbers (unless they share a factor).
    
    # Let's try a maximum clique approach using a greedy + local search strategy.
    
    # Build adjacency: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    n_cand = len(candidates)
    
    # For efficiency, let's work with the indices
    # First, let's get the baseline: all multiples of 3
    mult3 = set(range(3, 256, 3))
    baseline = len(mult3)  # 85
    
    # Try to improve by considering removing some elements from mult3 and adding others.
    # Elements that can potentially be added: those divisible by 5 or 17 but not 3.
    addable = set()
    for x in candidates:
        if x not in mult3 and x != target:
            # Check if x is divisible by 5 or 17
            if x % 5 == 0 or x % 17 == 0:
                addable.add(x)
    
    # Elements in mult3 that are "small" and might be removable:
    # If we remove 3, can we add numbers divisible by 5?
    # A number x div by 5 (not 3) needs gcd(x, y) > 1 for all y in (mult3 - {3}).
    # The smallest remaining multiple of 3 is 6. gcd(x, 6) > 1 means x shares 2 or 3 with 6.
    # x is not div by 3, so x must be even (div by 2).
    # Check: gcd(x, 9) > 1? x not div by 3, so need x div by 3... no. gcd(x,9)=1 unless 3|x.
    # So if 9 is in the set, x must share a factor with 9, i.e., x div by 3. Contradiction.
    
    # So removing just 3 isn't enough. We'd need to remove all multiples of 3 that don't share a factor with x.
    # This gets complex. Let's try a different approach.
    
    # Maximum clique via branch and bound on a reduced problem.
    # Since we have 129 candidates, this might be too large for brute force.
    # Let's use a heuristic approach.
    
    # Approach: Try different "strategies" based on which prime to focus on.
    
    best_set = set(mult3)
    best_size = baseline
    
    # Strategy 1: All multiples of 3 (baseline)
    # Strategy 2: Try to find a larger set by mixing.
    
    # Let's try: for each subset of "problematic" small multiples of 3 to remove,
    # check if we can add enough non-multiples-of-3 to compensate.
    
    # The problematic elements are those in mult3 that don't share a factor with a given candidate.
    # For a candidate x (div by 5, not 3): problematic = {y in mult3 : gcd(x,y) == 1}
    
    # Let's compute for each addable candidate, how many mult3 elements are incompatible.
    
    time_check = time.time()
    if time_check > deadline - 5:
        return sorted(best_set)
    
    # For each addable candidate, find incompatible elements in mult3
    incompat_map = {}
    for x in sorted(addable):
        incompat = set()
        for y in mult3:
            if math.gcd(x, y) == 1:
                incompat.add(y)
        incompat_map[x] = incompat
    
    # Now, if we want to add a set A of addable candidates, we need to remove
    # the union of incompat_map[x] for all x in A from mult3.
    # Net gain = |A| - |union of incompat sets|
    
    # Try single additions:
    for x in sorted(addable):
        if time.time() > deadline - 5:
            break
        incompat = incompat_map[x]
        net_gain = 1 - len(incompat)
        if net_gain > 0:
            new_set = (mult3 - incompat) | {x}
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set
    
    # Try pairs of additions:
    addable_list = sorted(addable)
    for i in range(len(addable_list)):
        if time.time() > deadline - 5:
            break
        x1 = addable_list[i]
        for j in range(i+1, len(addable_list)):
            x2 = addable_list[j]
            # Check if x1 and x2 are compatible
            if math.gcd(x1, x2) == 1:
                continue
            incompat_union = incompat_map[x1] | incompat_map[x2]
            net_gain = 2 - len(incompat_union)
            if net_gain > 0:
                new_set = (mult3 - incompat_union) | {x1, x2}
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
    
    # Try triples:
    for i in range(len(addable_list)):
        if time.time() > deadline - 5:
            break
        x1 = addable_list[i]
        for j in range(i+1, len(addable_list)):
            x2 = addable_list[j]
            if math.gcd(x1, x2) == 1:
                continue
            for k in range(j+1, len(addable_list)):
                x3 = addable_list[k]
                if math.gcd(x1, x3) == 1 or math.gcd(x2, x3) == 1:
                    continue
                incompat_union = incompat_map[x1] | incompat_map[x2] | incompat_map[x3]
                net_gain = 3 - len(incompat_union)
                if net_gain > 0:
                    new_set = (mult3 - incompat_union) | {x1, x2, x3}
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = new_set
    
    # Also try: start from multiples of 5 and see if we can do better
    mult5 = set(range(5, 256, 5))
    if len(mult5) > best_size:
        best_size = len(mult5)
        best_set = mult5
    
    # Try: all numbers divisible by 3 or 17 (but need pairwise gcd > 1)
    # This won't work simply because 3 and 5 are coprime... wait, 17 and 3 are coprime.
    # So we can't just union them.
    
    # Let's also try a local search: start with best_set and try swapping elements.
    if time.time() < deadline - 10:
        # Local search: try adding one element and removing one or more
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for x in candidates:
                if x in best_set:
                    continue
                # Check if x is compatible with all elements in best_set
                compatible = all(math.gcd(x, y) > 1 for y in best_set)
                if compatible:
                    best_set.add(x)
                    best_size += 1
                    changed = True
                    break
    
    # Verify the result
    result = sorted(best_set)
    # Ensure 255 is in the set
    if target not in result:
        result.append(target)
        result = sorted(result)
    
    # Final verification
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # Fallback to multiples of 3
                return list(range(3, 256, 3))
    
    return result
# EVOLVE-BLOCK-END
