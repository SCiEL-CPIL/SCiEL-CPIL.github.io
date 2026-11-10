# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 78  # 78s margin out of 90s
    n_max = 512
    
    # Precompute for the full range [1, 512]
    # We maintain a dynamic programming state as we add elements.
    # `sums_by_size[k]` is a set of sums of subsets of size `k` from the current set.
    # Also, we might need `sums_excl_elem` or similar for the reverse check?
    # Let's look at the condition:
    # No element `a` in the set `S` is the mean of any subset `T` of `S \ {a}` with `|T| >= 2`.
    # This is equivalent to:
    # For every `a` in `S`, and every `k >= 2`, `k * a` is NOT in `sums_by_size[k]` of `S \ {a}`.
    
    # When we add a new element `x` to `S` to form `S' = S U {x}`:
    # 1. Check if `x` is a valid mean in `S`.
    #    i.e., for all `k >= 2`, `k * x` is NOT in `sums_by_size[k]` of `S`.
    #    This is easy to check with the existing `sums_by_size` of `S`.
    
    # 2. Check if any existing `a` in `S` becomes an invalid mean in `S'`.
    #    This happens if `a` is the mean of a subset `T` in `S'` with `|T| >= 2`.
    #    Since `S` was valid, `T` cannot be a subset of `S`.
    #    So `T` must contain `x`.
    #    Let `T = U U {x}`, where `U` is a subset of `S \ {a}`.
    #    Then `mean(T) = a` implies `sum(U) + x = (|U| + 1) * a`.
    #    `sum(U) = (|U| + 1) * a - x`.
    #    Let `j = |U|`. Then `j` can range from 1 to `|S|-1` (since `a` is in `S`, and `U` is from `S\{a}`).
    #    Wait, `T` has size `j+1`. The condition is `|T| >= 2`, so `j+1 >= 2 => j >= 1`.
    #    So we need to check if there exists `a in S` and `j >= 1` such that
    #    `(j+1)*a - x` is a sum of `j` elements from `S \ {a}`.
    
    # This second check is expensive if done naively.
    # However, note that `sums_by_size` is for the whole set `S`.
    # We need sums from `S \ {a}`.
    # We can maintain `sums_by_size_excl[a][k]`? That would be `O(N * K * S)` memory which is too high.
    
    # Alternative:
    # We only add `x` if it's valid.
    # The "existing element becomes invalid" check is the bottleneck.
    # Is it possible that adding `x` makes an existing `a` invalid?
    # Yes. Example: S={1, 3}. 2 is not in S.
    # If we add 2, S'={1,2,3}. 2 is mean of {1,3}. But 1 and 3 were in S.
    # Was S valid?
    # For S={1,3}:
    # 1 is not mean of subset of {3} (size 1, but we need size >= 2? No, "subset of at least two OTHER elements").
    # So for S={1,3}, are there any elements that are means of subsets of size >= 2 of others?
    # Others for 1: {3}. Size 1. No subset of size >= 2. So 1 is safe.
    # Others for 3: {1}. Size 1. No subset of size >= 2. So 3 is safe.
    # So S={1,3} is valid.
    # If we add 2:
    # Check 1: Is 2 mean of subset of {1,3} of size >= 2?
    # Subset {1,3} has sum 4. 4/2 = 2. Yes.
    # So Check 1 fails. 2 cannot be added.
    
    # Let's try a case where Check 1 passes but Check 2 fails.
    # S = {1, 4, 6}.
    # Is S valid?
    # 1: others {4,6}. Subset {4,6} sum 10, mean 5 != 1.
    # 4: others {1,6}. Subset {1,6} sum 7, mean 3.5 != 4.
    # 6: others {1,4}. Subset {1,4} sum 5, mean 2.5 != 6.
    # So S is valid.
    # Add x = 5.
    # Check 1: Is 5 mean of subset of {1,4,6}?
    # Subsets of size 2:
    # {1,4} sum 5, mean 2.5
    # {1,6} sum 7, mean 3.5
    # {4,6} sum 10, mean 5. YES.
    # So Check 1 fails.
    
    # It seems Check 1 is quite strong.
    # What if the subset that makes `a` invalid uses `x` and some elements, but the total size is large?
    # The condition is `sum(U) + x = (j+1)a`.
    # This implies `x = (j+1)a - sum(U)`.
    # Since `U` is a subset of `S`, `sum(U)` is a known sum.
    # So for a fixed `a` and `j`, `x` is determined if `sum(U)` is determined.
    # But we are iterating over `x` to find a candidate.
    
    # Let's optimize the check.
    # We can precompute `sums_by_size` for the current set `S`.
    # For Check 1, we just look up `k*x` in `sums_by_size[k]`.
    # For Check 2, we need to know if `(j+1)a - x` is a sum of `j` elements from `S \ {a}`.
    # Note that `S \ {a}` is `S` minus one element.
    # `sums_by_size_excl[a][j]` would be ideal.
    # But we can approximate or use a different strategy.
    
    # Strategy:
    # Since `N=512` is small, and we want max size ~48, the set `S` grows slowly.
    # We can afford to do a somewhat expensive check for each candidate `x`.
    # However, `S \ {a}` changes for each `a`.
    
    # Let's try to maintain `sums_by_size` for `S`.
    # And for Check 2, we can iterate `a in S`.
    # For each `a`, we need to check if there is a `j` and a sum `V = (j+1)a - x` such that `V` is a sum of `j` elements from `S \ {a}`.
    # We can compute `sums_by_size_excl[a]` on the fly?
    # No, that's too slow.
    
    # Alternative idea:
    # The condition "a is mean of subset of size >= 2 of others" is equivalent to:
    # `a` is NOT in the set of means of all subsets of `S \ {a}` of size >= 2.
    # Let `M(S)` be the set of all means of subsets of `S` of size >= 2.
    # Then `S` is valid iff `S` is disjoint from `M(S \ {a})` for all `a`? No.
    # `S` is valid iff for all `a in S`, `a` is not in `M(S \ {a})`.
    
    # When adding `x`:
    # 1. `x` not in `M(S)`.
    # 2. For all `a in S`, `a` not in `M(S' \ {a})`.
    #    `S' \ {a} = (S \ {a}) U {x}`.
    #    `M((S \ {a}) U {x})` includes means of subsets of `S \ {a}` (already checked for `S` valid? No, `S` valid means `a` not in `M(S \ {a})`. But now we have `x` added).
    #    The new means are those involving `x`.
    #    Mean of `U U {x}` where `U` subset of `S \ {a}`.
    #    `mean = (sum(U) + x) / (|U| + 1)`.
    #    We need this not to be `a`.
    #    `sum(U) + x != (|U| + 1) a`.
    #    `sum(U) != (|U| + 1) a - x`.
    
    # So for each `a in S`, we need to ensure that for all `j >= 1` (where `j = |U|`),
    # `(j+1)a - x` is NOT a sum of `j` elements from `S \ {a}`.
    
    # How to check this fast?
    # We can precompute `sums_by_size_excl[a][j]` for all `a` and `j`?
    # Number of `a` is up to 48. Number of `j` up to 47.
    # Total sets: 48 * 47 ~ 2200 sets.
    # Each set can have up to `C(47, 23)` sums? No, the sums are bounded by `47 * 512 ~ 24000`.
    # So we can use bitsets or boolean arrays for the sums.
    # But maintaining these for every addition might be costly.
    
    # Let's try a simpler approach first:
    # Just use `sums_by_size` for `S`.
    # For Check 2, we can iterate `a`.
    # We know `sums_by_size` for `S`.
    # `S \ {a}` is `S` without `a`.
    # Can we derive `sums_by_size_excl[a]` from `sums_by_size`?
    # Not easily, because removing an element affects all subset sums.
    
    # However, notice that we only add `x` if it passes.
    # The number of elements in `S` is small (~48).
    # The number of candidates `x` is 512.
    # Total checks: 512 * 48 * 47 * (cost of subset sum check).
    # If we use a DP for subset sum for each `(a, j)`, it might be too slow.
    
    # Let's use a different perspective.
    # The condition is symmetric.
    # Maybe we can just check `x` against the current set `S` and assume that adding `x` won't break existing elements?
    # Is that true?
    # If `S` is valid, and `x` is not a mean of any subset of `S` of size >= 2, is `S' = S U {x}` valid?
    # No, as shown in my hypothetical (though I couldn't find a concrete counter-example easily, it's theoretically possible).
    # But in practice, for "mean avoiding" sets, the greedy approach often works well.
    # Let's try the greedy approach with only Check 1 and see the score.
    # If it's low, we add Check 2.
    
    # Actually, let's look at the constraint again.
    # "No element may be the exact arithmetic mean of any subset of at least two OTHER elements."
    
    # Let's implement the greedy with Check 1 first. It's very fast.
    # If we reach 48, we return.
    
    sums_by_size = {0: {0}}
    selected = []
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        
        valid = True
        # Check 1: x is not mean of any subset of selected
        for k in range(2, len(selected) + 1):
            if k * x in sums_by_size.get(k, set()):
                valid = False
                break
        
        if not valid:
            continue
        
        # If we want to be strict, we should do Check 2.
        # Let's try to do Check 2 efficiently.
        # We can maintain `sums_excl`?
        # Or just do a DP for `selected` without `a`?
        # Since `len(selected)` is small, we can do:
        for a in selected:
            # We need to check if a is mean of any subset of (selected + [x]) containing x
            # which is equivalent to:
            # sum(U) + x = (len(U) + 1) * a for some U subset of selected \ {a}
            # sum(U) = (len(U) + 1) * a - x
            # Let's create a list `others = [y for y in selected if y != a]`
            others = [y for y in selected if y != a]
            # DP for subset sums of others
            # max_size = len(others)
            # dp[j] = set of sums of size j
            # But we only care about specific targets.
            # Target for size j is T_j = (j+1)*a - x
            # We can do a backward DP or just iterate subsets if others is small?
            # len(others) <= 47.
            # DP is O(N * K). 47 * 47 ~ 2200 operations.
            # For each a, this is fast.
            # Total for Check 2: 48 * 2200 ~ 100k operations per candidate x.
            # 512 candidates * 100k = 50 million operations.
            # In Python, this might take a few seconds.
            # Let's try it.
            
            # Optimization: Precompute min/max sums for others?
            # Or just do the DP.
            
            # To make it faster, we can use a bitset for the sums?
            # Sums are up to 47 * 512 = 24064.
            # We can use an integer as a bitset.
            # bit i is 1 if sum i is possible.
            
            # Let's use Python integers as bitsets for the DP.
            # dp[j] is an integer where bit s is 1 if sum s is possible with j elements.
            
            # Initialize dp
            # dp[0] = 1 (bit 0 is 1)
            # For each val in others:
            #   for j from current_max_size down to 1:
            #      dp[j] |= dp[j-1] << val
            
            # Then check if bit T_j is set in dp[j].
            
            dp = [0] * (len(others) + 1)
            dp[0] = 1  # bit 0 set
            current_max_j = 0
            for val in others:
                # Update dp from back to front
                # We only need to go up to len(others)
                for j in range(current_max_j, 0, -1):
                    dp[j] |= (dp[j-1] << val)
                dp[1] |= (1 << val) # Wait, dp[0] is 1. dp[1] |= dp[0] << val.
                # The loop above handles j=1 if current_max_j >= 1.
                # Let's re-structure.
                
                # Standard 0/1 knapsack for counting size.
                # We need to be careful with the order.
                
                # Let's just use sets for simplicity first, and optimize if it's too slow.
                # Given the deadline, let's try the set-based DP first.
                
            # Actually, the bitset approach with Python integers is very fast.
            # Let's implement it correctly.
            
            # Reset dp
            dp = [0] * (len(others) + 1)
            dp[0] = 1
            # We process elements one by one.
            for val in others:
                # Update j from high to low
                # max j for this val is the number of elements processed so far + 1
                # But we can just loop from len(others) down to 1
                # It's safe because dp[j-1] from the previous step is used.
                # Wait, if we update in place, we must go downwards.
                for j in range(len(others), 0, -1):
                    if dp[j-1]:
                        dp[j] |= (dp[j-1] << val)
                        
            # Now dp[j] contains all subset sums of size j from `others`.
            # Check targets
            for j in range(1, len(others) + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0 or target_sum > 25000: # Bound check
                    continue
                # Check if bit target_sum is set in dp[j]
                if (dp[j] >> target_sum) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            # Update sums_by_size for Check 1
            # sums_by_size[k] |= {s + x for s in sums_by_size[k-1]}
            new_sums = {k: set() for k in range(len(selected) + 1)}
            new_sums[0] = {0}
            for k in range(1, len(selected) + 1):
                if k in sums_by_size:
                    new_sums[k].update(sums_by_size[k])
                if k-1 in sums_by_size:
                    new_sums[k].update(s + x for s in sums_by_size[k-1])
            sums_by_size = new_sums
            
            if len(selected) >= 48:
                break
                
    return selected
# EVOLVE-BLOCK-END
