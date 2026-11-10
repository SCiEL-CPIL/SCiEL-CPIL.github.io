import time
import random

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(arr):
    s = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            sm = a + b
            if sm in s:
                return False
            s.add(sm)
    return True

def _is_maximal(arr, used_sums):
    arr_set = set(arr)
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        # check if x can be added
        ok = True
        for a in arr:
            sm = x + a
            if sm in used_sums:
                ok = False
                break
        if ok:
            if 2 * x in used_sums:
                ok = False
        if ok:
            return False
    return True

def _compute_sums(arr):
    s = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            s.add(a + arr[j])
    return s

def _greedy_descending(rng):
    # Descending greedy: try to add x if it doesn't conflict with existing sums
    selected = []
    sums = set()
    # Randomize the order slightly by shuffling a copy? 
    # Actually, pure descending is a specific strategy. 
    # Let's try random permutations of the range, or descending with random tie-breaking?
    # The prompt says "randomized greedy". 
    # Let's stick to a randomized order, but bias towards larger numbers?
    # Or just random order?
    # Descending greedy often produces smaller maximal sets because large numbers have fewer "room" 
    # for conflicts with each other? No, large numbers have large sums, so they don't conflict 
    # with small numbers' sums as easily? 
    # Actually, small numbers create many small sums. Large numbers create large sums.
    # If we pick large numbers first, the sums are large. 
    # If we then pick small numbers, their sums are small, so they don't conflict with the large sums.
    # This allows packing more numbers? Or fewer?
    # Maximal means we CAN'T add any more. 
    # If we have a set of large numbers, can we add a small number? 
    # The small number x conflicts if x+a is in sums. 
    # Since sums are large, x+a is likely not in sums. 
    # So we might be able to add MANY small numbers. 
    # This suggests descending greedy might result in a LARGER set.
    
    # Let's try RANDOM order.
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        added_sums = [x + a for a in selected] + [2 * x]
        # Check distinctness within added
        if len(set(added_sums)) != len(added_sums):
            continue
        # Check conflict with existing
        if sums.intersection(added_sums):
            continue
        selected.append(x)
        sums.update(added_sums)
    
    return selected

def _try_remove(arr, idx, sums):
    # Check if removing arr[idx] still yields a maximal Sidon set
    # 1. New set is Sidon (it is, subset of Sidon)
    # 2. New set is Maximal
    # To check maximality, we need to know if any missing element can be added.
    # This is expensive. 
    # Optimization: 
    # If we remove x, the set becomes S' = S \ {x}.
    # S' is maximal if for all y not in S', y cannot be added to S'.
    # Note that x itself is now missing. Can x be added back to S'? 
    # Yes, trivially, because S was Sidon. So if we just remove x, x can be added back.
    # Thus S \ {x} is NEVER maximal (unless N=1 or something trivial, but here N=256).
    # Wait. The definition of maximal Sidon set: cannot be extended by ANY element from [1,N].
    # If I remove x, then x is a missing element. 
    # Can I add x back? 
    # The sums of S \ {x} are a subset of sums of S. 
    # When adding x, the new sums are {x+a for a in S\{x}} U {2x}.
    # These were all present in S's sums (since x was in S). 
    # So x CAN be added back. 
    # Therefore, simply removing an element ALWAYS breaks maximality.
    
    # This implies that "element removal" as a local search move is INVALID for maintaining maximality 
    # UNLESS we also add another element or modify the set such that the removed element can no longer be added back.
    
    # How can we prevent x from being added back?
    # We need to ensure that for the new set S', x+a is in sums(S') or 2x is in sums(S') for some a in S'.
    # But sums(S') is sums(S) minus {x+a for a in S} and {2x}.
    # So we need some other pair in S' to produce a sum that conflicts with x.
    
    # This suggests a "swap" move or a more complex move is needed.
    # Or, we can remove x and ADD y, such that y blocks x.
    
    # Given the complexity, let's stick to the observed strategy in the prompt:
    # "Randomized greedy generation combined with aggressive local search (element removal) to minimize cardinality... Measured: 1 -> 1.143"
    # The prompt claims this worked. How?
    # Maybe the "removal" isn't just deleting? 
    # Or maybe my analysis is wrong?
    # "Adding ANY missing integer from [1,N] must destroy this property."
    # If I have S, and I remove x to get S'. 
    # x is missing from S'. 
    # Can I add x to S'? 
    # Yes, because S was Sidon. 
    # So S' is NOT maximal. 
    # So the score would be 0 (invalid).
    
    # Unless... the "local search" involves checking if the set is maximal, and if not, 
    # it tries to "repair" it by adding elements? 
    # But adding elements INCREASES cardinality. We want to MINIMIZE cardinality.
    
    # Is it possible that the "removal" strategy refers to removing elements from a NON-maximal set 
    # to make it maximal? No, removing elements makes it LESS maximal.
    
    # Let's re-read the prompt's insight carefully:
    # "Randomized greedy generation combined with aggressive local search (element removal) to minimize cardinality of maximal Sidon sets, exploiting the fact that removal is cheap to verify for maximality."
    
    # This is confusing. Removal is NOT cheap to verify for maximality (it's O(N^2) or O(N*|S|)).
    # And removal breaks maximality.
    
    # Perhaps the strategy is:
    # 1. Generate a RANDOM Sidon set (not necessarily maximal).
    # 2. Make it maximal by ADDING elements (greedy saturation). This increases size.
    # 3. Then try to REMOVE elements? But that breaks maximality.
    
    # Maybe the "removal" is from the list of CANDIDATES? No.
    
    # Let's look at the score 1.143. The seed is 1.0. 
    # If the seed returns a valid maximal set, its score is 1.0 (seed/value, value=1? No, seed program score is 1.0).
    # The seed program returns a greedy maximal set. 
    # If the seed program returns a set of size K, and the "true" minimal maximal size is K_min, 
    # and the score is seed/value... 
    # "The evaluator measures the raw value (lower is better). The search score is seed/value"
    # Seed program score is 1.0. So Seed_Value / Value = 1.0? 
    # Or is Score = Seed_Size / Current_Size? 
    # If Seed_Size = 20 and Current_Size = 17, Score = 20/17 = 1.17.
    # This matches "1 -> 1.143" if the size decreased.
    
    # So the goal is to find a SMALLER maximal Sidon set than the greedy one.
    
    # How can we find a smaller maximal set?
    # The greedy set is maximal. 
    # Any subset of a maximal set is NOT maximal.
    # So we cannot just remove elements from the greedy set.
    
    # We must find a DIFFERENT set that is maximal and smaller.
    # This requires a global search or a smarter construction.
    
    # Strategy: 
    # 1. Randomized Greedy (as above) to get a maximal set S1.
    # 2. Randomized Greedy again to get S2.
    # 3. Keep the smallest one.
    # 4. Try "Swaps": Remove x, add y. 
    #    If we remove x and add y, the size stays the same. 
    #    We want to REDUCE size.
    #    So we need a move that reduces size.
    #    But any move that reduces size (removal) breaks maximality.
    #    So we must ADD elements to restore maximality? 
    #    If we remove x (size -1), we must add k elements to restore maximality.
    #    If k=0, size decreases. But k=0 is impossible as x can be added back.
    #    If k=1, we remove x and add y. Size same.
    #    If we can find a sequence of swaps that allows a removal?
    
    # Actually, there is a known heuristic for "minimal maximal" sets:
    # Start with a random set. 
    # Make it maximal (add elements).
    # Then, try to remove elements. If removing x breaks maximality, it's because some y can be added.
    # If the "blocking" y is x itself, we are stuck.
    # But if the set was constructed differently, maybe x is blocked by other elements?
    # No, if x is in the set, it's not "missing".
    # If we remove x, x becomes missing. x is always addable to S\{x}.
    # So S\{x} is NEVER maximal.
    
    # Conclusion: You CANNOT reduce the size of a maximal set by simply removing elements.
    # You must find a completely different maximal set that happens to be smaller.
    
    # So the "local search" must be exploring the space of maximal sets via swaps, 
    # hoping to find a "basin" with smaller size? 
    # But swaps preserve size.
    
    # Maybe the "removal" insight is misleading or I am missing a nuance.
    # "exploiting the fact that removal is cheap to verify for maximality"
    # Maybe it means: verify if a set IS maximal by trying to add elements?
    # No, "removal".
    
    # Let's assume the prompt's insight is: 
    # "Use randomized greedy to generate diverse maximal sets. Pick the smallest."
    # This is a valid strategy. Different random orders lead to different maximal sets of different sizes.
    # The greedy maximal set size varies with the order.
    # Descending vs Ascending vs Random can yield different sizes.
    # We will run multiple random greedy passes and keep the best (smallest) one.
    
    # Additionally, we can try to "shrink" a set by:
    # 1. Removing an element x.
    # 2. The set is no longer maximal. 
    # 3. Identify ALL elements that can now be added.
    # 4. If the ONLY element that can be added is x, we are stuck.
    # 5. If we can add a DIFFERENT element y, we can swap x for y.
    # 6. This doesn't reduce size.
    
    # Is it possible to have a maximal set where removing x allows adding y, and then removing y allows adding z... and eventually we can remove an element and add NONE?
    # No, because the removed element is always addable.
    
    # So the ONLY way to get a smaller set is to find a different maximal set from scratch.
    
    # Strategy:
    # 1. Run many randomized greedy constructions.
    # 2. Keep the one with minimum cardinality.
    # 3. (Optional) Try to improve via "2-opt" style swaps if time permits, though swaps don't reduce size.
    #    Wait, if I have a set S of size K.
    #    Can I find a maximal set S' of size K-1?
    #    Yes, if such a set exists.
    #    How to find it? Random search.
    
    # Let's implement:
    # - Deadline guard.
    # - Loop:
    #   - Generate random order.
    #   - Greedy saturation.
    #   - Check size.
    #   - Update best.
    
    # To make it better than the seed (which is deterministic ascending), 
    # randomization is key. Ascending greedy might produce a "bad" (large) maximal set.
    # Descending might be better? Or random?
    # Let's try random.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Initial deterministic descending to get a baseline
    # Descending greedy
    selected = []
    sums = set()
    for x in range(N, 0, -1):
        added_sums = [x + a for a in selected] + [2 * x]
        if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
            selected.append(x)
            sums.update(added_sums)
    if len(selected) < best_size:
        best_size = len(selected)
        best_set = selected[:]
        
    # Randomized greedy loop
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
                selected.append(x)
                sums.update(added_sums)
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            # If we found a very small set, we might break early?
            # Theoretical minimum is ~sqrt(N) ~ 16.
            # If we get below 20, it's good.
            if best_size < 18:
                break
                
    return best_set
# EVOLVE-BLOCK-END
