import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all
    distinct pairs. Maximize cardinality.
    
    Strategy:
    255 = 3 * 5 * 17.
    
    The seed returns multiples of 5: {5, 10, ..., 255} → 51 elements.
    
    A simple improvement: all multiples of 3 in [1,255] → 85 elements, all pairwise
    share factor 3, and 255 is included. Score = 85/51 ≈ 1.667.
    
    Can we do better? We want the largest subset S ⊆ [1,255] with 255 ∈ S and
    gcd(a,b) > 1 for all distinct a,b ∈ S.
    
    Key insight: If we take all multiples of 3, we get 85. But we might be able to
    add some numbers that aren't multiples of 3, as long as they share a factor with
    every other element. However, if we add a number x that is NOT a multiple of 3,
    then x must share a factor with every multiple of 3 in S. The multiples of 3
    include numbers like 3, 6, 9, ... For x to share a factor with 3, x must be
    divisible by 3 (contradiction) — wait, gcd(x, 3) > 1 means x is divisible by 3.
    So actually, any x added to the set of ALL multiples of 3 must itself be a multiple
    of 3 (since gcd(x,3) must be > 1). So the set of all multiples of 3 is maximal
    in the sense that you can't add anything else.
    
    But what if we DON'T take all multiples of 3? What if we use a different strategy?
    
    Alternative: Take all multiples of 3 OR 5 (i.e., numbers divisible by 3 or 5).
    But then two numbers, one divisible only by 3 and one only by 5, would have gcd 1.
    So that doesn't work.
    
    What about: all numbers divisible by 3, plus some numbers divisible by 5 (but not 3)
    that also share factors with all multiples of 3? As argued above, a number not
    divisible by 3 cannot share a factor with 3, so it can't be added.
    
    Hmm, but what if we remove 3 from the set? Then we could potentially add numbers
    not divisible by 3. Let's think more carefully.
    
    Actually, let's think about this differently. The problem is equivalent to finding
    the largest clique in a graph where vertices are [1,255] and edges connect pairs
    with gcd > 1, with the constraint that 255 must be in the clique.
    
    Since 255 = 3*5*17, any element in the clique must share a factor with 255, so
    every element must be divisible by 3, 5, or 17.
    
    Now, among elements divisible by 3, 5, or 17, we need all pairs to have gcd > 1.
    
    Let's consider: can we take all multiples of 3 (85 elements) plus some multiples
    of 5 or 17 that aren't multiples of 3?
    
    If we add x (multiple of 5, not of 3), then for every y in our set (multiples of 3),
    we need gcd(x,y) > 1. Since x is not divisible by 3, gcd(x,y) > 1 requires y to
    share another factor with x. But y ranges over all multiples of 3, including 3 itself.
    gcd(x, 3) = 1 since x is not divisible by 3. So we can't add any non-multiple-of-3
    if 3 is in the set.
    
    What if we remove 3? Then we have multiples of 3 except 3: {6, 9, 12, ..., 255}.
    Can we add a multiple of 5 not divisible by 3, say 5? gcd(5, 6) = 1. No.
    Say 10? gcd(10, 9) = 1. No.
    
    It seems hard to add non-multiples of 3.
    
    What about using 5 as the common factor instead? Multiples of 5: 51 elements. Worse.
    
    What about 17? Multiples of 17: 15 elements. Much worse.
    
    So 85 seems like a strong candidate. But let's think if there's a smarter construction.
    
    What if we take all numbers in [1,255] that are divisible by at least two of {3,5,17}?
    No, that's too restrictive.
    
    What about: all multiples of 3 (85 elements). This gives score 85/51 ≈ 1.667.
    
    Can we do even better? Let's think about whether we can construct a set larger than 85.
    
    The maximum possible size: every element must share a factor with 255, so every element
    is divisible by 3, 5, or 17. Count of such numbers in [1,255]:
    - Divisible by 3: 85
    - Divisible by 5: 51
    - Divisible by 17: 15
    - Divisible by 3 and 5: 17
    - Divisible by 3 and 17: 5
    - Divisible by 5 and 17: 3
    - Divisible by 3, 5, 17: 1
    
    By inclusion-exclusion: 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127.
    
    So there are 127 candidates. But we need a clique among them.
    
    The set of all multiples of 3 is a clique of size 85. Can we find a larger clique?
    
    Let's try a greedy/local search approach to see if we can find a clique larger than 85.
    Given the 60-second budget, we can afford some search.
    
    Actually, let me think about this more carefully. The key constraint is that for any
    two elements a, b in our set, gcd(a,b) > 1. 
    
    Consider the set of all multiples of 3. Size 85. Now, can we replace some multiples
    of 3 with non-multiples of 3 to increase the size? Unlikely, since we'd need to
    maintain the pairwise gcd property.
    
    Let me try a different approach: what if we use a set where all elements share a
    common prime factor? The best is multiples of 3 (85 elements).
    
    But what if we use a "union" strategy? For example, take all multiples of 3 AND
    all multiples of 5, but ensure pairwise gcd > 1. The problem is that a multiple of 3
    (not 5) and a multiple of 5 (not 3) may have gcd 1.
    
    I think 85 is likely optimal or near-optimal. Let me verify with a quick check:
    Is there any clique larger than 85 in this graph?
    
    Given the computational budget, let me implement a heuristic search to try to find
    a clique larger than 85, but default to 85 if the search doesn't find better.
    """
    deadline = time.time() + 85  # 85 seconds, leaving 5s margin
    
    # 255 = 3 * 5 * 17
    # All elements must share a factor with 255, so must be divisible by 3, 5, or 17.
    
    # Baseline: all multiples of 3
    baseline = list(range(3, 256, 3))
    baseline_size = len(baseline)  # 85
    
    # Try to find a larger clique using local search
    # Candidates: all numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    
    # Precompute adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Start with the set of all multiples of 3
    start_set = set()
    for i, v in enumerate(candidates):
        if v % 3 == 0:
            start_set.add(i)
    
    best_set = set(start_set)
    best_size = len(best_set)
    
    # Local search: try adding one element, removing another
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        # Try adding a random candidate not in the set
        not_in_set = [i for i in range(n) if i not in best_set]
        if not not_in_set:
            break
        # Try removing a random element from the set
        in_set = list(best_set)
        
        # Try add-only
        for _ in range(20):
            if time.time() > deadline:
                break
            idx = not_in_set[int(time.time() * 1000000) % len(not_in_set)]
            # Check if idx can be added: must be adjacent to all in best_set
            can_add = all(j in adj[idx] for j in best_set)
            if can_add:
                new_set = best_set | {idx}
                if len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    not_in_set = [i for i in range(n) if i not in best_set]
        
        if improved:
            continue
        
        # Try swap: remove one, add one
        for _ in range(10):
            if time.time() > deadline:
                break
            if not in_set or not not_in_set:
                break
            rem = in_set[int(time.time() * 777777) % len(in_set)]
            add = not_in_set[int(time.time() * 123456) % len(not_in_set)]
            new_set = (best_set - {rem}) | {add}
            # Verify pairwise gcd
            valid = True
            vals = [candidates[i] for i in new_set]
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    if math.gcd(vals[i], vals[j]) == 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(new_set) > best_size:
                best_set = new_set
                best_size = len(best_set)
                in_set = list(best_set)
                not_in_set = [i for i in range(n) if i not in best_set]
                improved = True
    
    # Convert back to values
    result = [candidates[i] for i in best_set]
    result.sort()
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
# EVOLVE-BLOCK-END
