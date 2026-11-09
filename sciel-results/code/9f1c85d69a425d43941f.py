"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def construct():
    """
    Find the largest subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    
    Uses backtracking with forward checking. We process elements in a specific order
    and maintain the set of 'forbidden sums' that would conflict with already-selected
    elements.
    """
    N = 24
    elements = list(range(1, N + 1))
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recips = {x: Fraction(1, x) for x in elements}
    
    # We want to maximize cardinality. 
    # Strategy: Try to include as many elements as possible.
    # A key insight: if we pick a set S, for each x in S, 1/x must not be in the set of sums of subsets of S\{x}.
    # This is equivalent to: 1/x is not a "subsum" of S\{x}.
    
    # Backtracking approach:
    # We build the set incrementally. To make it efficient, we should probably try to include larger numbers first?
    # Or smaller? 
    # Let's think about conflicts. 
    # 1/1 = 1. Can 1 be a sum of reciprocals of others? Yes, e.g., 1/2+1/3+1/6=1. So if 1 is in S, none of its "subsum-free" subsets can be in S.
    # Actually, the condition is: for each x in S, 1/x is NOT in { sum(1/y for y in T) | T subset of S\{x}, T non-empty }.
    
    # Let's use a recursive backtracking that tries to add elements one by one.
    # To maximize size, we can iterate from largest possible size down, or just do a DFS that prefers adding elements.
    # Since N=24 is small, a well-pruned DFS should work.
    
    # Order of processing: 
    # If we process in increasing order, we might add small numbers which create many constraints.
    # If we process in decreasing order, large numbers have small reciprocals, which are less likely to be sums of other reciprocals?
    # Actually, sums of reciprocals of large numbers are small.
    # Let's try processing in increasing order of the element value (1 to 24).
    # Or maybe decreasing? 
    # Let's stick to a standard DFS with pruning.
    
    # State:
    # current_set: list of selected elements
    # index: next element to consider (if we go in fixed order)
    # 
    # However, the constraint is global. When we add a new element x to S, we must check:
    # 1. For the new x: 1/x is not a sum of any subset of S_old.
    # 2. For every old element y in S_old: 1/y is not a sum of any subset of S_new\{y}.
    #    Note: S_new\{y} = (S_old\{y}) U {x}.
    #    So we need to check if 1/y can be formed by a subset of (S_old\{y}) U {x}.
    #    This is equivalent to: 1/y is a sum of a subset of S_old\{y} (which was already false) OR
    #    1/y is a sum of {x} U T where T is a subset of S_old\{y}.
    #    i.e., 1/y - 1/x is a sum of a subset of S_old\{y}.
    
    # This suggests we need to maintain the set of all possible sums of reciprocals of the current set.
    # Let Sums(S) be the set of all sums of non-empty subsets of S.
    # Condition for S to be valid: for all x in S, 1/x not in Sums(S\{x}).
    # Note that Sums(S\{x}) = Sums(S) \ { sums involving x } ... not quite, it's the set of sums of subsets of S that do not contain x.
    
    # Let's maintain `sums` as the set of all possible sums of non-empty subsets of the current selected set.
    # When we add x:
    # new_sums = sums U { s + 1/x for s in sums } U { 1/x }
    # But we also need to check validity.
    # 
    # Let's refine the check:
    # Let S be the current set. Let R = { 1/y for y in S }.
    # Let SumSet be the set of all sums of non-empty subsets of R.
    # Validity condition: for each r in R, r is NOT in SumSet \ { sums that include r }.
    # Actually, it's simpler: for each x in S, 1/x is not in Sums(S \ {x}).
    
    # In backtracking:
    # We have a current set S and its corresponding SumSet(S).
    # We consider adding x.
    # Let S' = S U {x}.
    # We need to check:
    # 1. 1/x not in Sums(S). (Since Sums(S\{x}) for the new x is just Sums(S))
    # 2. For each y in S: 1/y not in Sums(S' \ {y}).
    #    S' \ {y} = (S \ {y}) U {x}.
    #    Sums(S' \ {y}) = Sums(S \ {y}) U { 1/x + s for s in Sums(S \ {y}) } U { 1/x }
    #    We know 1/y is not in Sums(S \ {y}) (by induction).
    #    So we just need to check if 1/y is in { 1/x + s for s in Sums(S \ {y}) } U { 1/x }.
    #    i.e., 1/y == 1/x  => y=x (impossible since distinct)
    #    or 1/y - 1/x == s for some s in Sums(S \ {y}).
    
    # This is getting complicated to maintain incrementally without recomputing Sums(S\{y}) for all y.
    # Given N=24, the size of SumSet can be large (exponential in worst case, but bounded by value range).
    # Reciprocals are Fractions. The denominators grow.
    # However, the sums are bounded by sum(1/i for i in 1..24) which is small (~3.8).
    # The number of distinct sums might be manageable.
    
    # Alternative: Just recompute the validity of the entire set S' when we add x?
    # Size of S is at most 24. Number of subsets is 2^24 ~ 16M. Too slow if done naively for each step.
    # But we can prune heavily.
    
    # Let's try a different angle:
    # The condition "1/x is not a subsum of S\{x}" is equivalent to saying that in the set of reciprocals R = {1/x for x in S}, no element is a sum of a subset of the other elements.
    # This is the definition of a "sum-free" set in the additive group of positive rationals? No, just that no element is a sum of others.
    # This is a "complete sequence" property inverse? No.
    # It's simply that the set R is "subsum-free".
    
    # Let's use a greedy heuristic with local search?
    # Or simple backtracking with a good ordering.
    
    # Let's try backtracking where we decide for each element (from 1 to 24) whether to include it or not.
    # We maintain the current set S and the set of all subset sums of reciprocals of S.
    # To check if we can add x:
    #   1. Check if 1/x is in Sums(S). If yes, cannot add.
    #   2. Check for all y in S: is 1/y in Sums(S U {x}) \ {y}?
    #      Sums(S U {x}) \ {y} consists of sums from S\{y} and sums from S\{y} + 1/x.
    #      We know 1/y is not in Sums(S\{y}).
    #      So we check if 1/y - 1/x is in Sums(S\{y}).
    
    # To make this efficient, we can precompute for the current S, for each y in S, the set Sums(S\{y}).
    # But that's O(|S| * 2^|S|) memory/time per step.
    
    # Given the small N (24), let's just try a DFS that builds the set and checks validity at the end?
    # No, we need pruning.
    
    # Let's try a simpler approach:
    # Iterate through all subsets? 2^24 is too big.
    # 
    # Let's use the property that if we sort the elements, maybe we can prune.
    # 
    # Actually, let's look at the structure.
    # 1/1 = 1. 1/2+1/3+1/6 = 1. So {1,2,3,6} is invalid because 1/1 = 1/2+1/3+1/6.
    # 
    # Let's write a function `is_valid(S)` that checks the condition.
    # Then use a heuristic search to find a large valid S.
    
    # Heuristic: Start with all elements, remove those that cause conflicts.
    # Or start with empty, add elements that don't conflict.
    
    # Let's try a greedy approach:
    # 1. Start with S = [].
    # 2. For each x in 1..24 (in some order):
    #    Try to add x. If valid, add it.
    # 3. To maximize size, we might need to try different orders or use local search.
    
    # Let's implement `is_valid(S)` efficiently.
    # For each x in S, compute all subset sums of S\{x}. Check if 1/x is in there.
    # |S| <= 24. 2^23 ~ 8M subsets. If we do this for 24 elements, 24 * 8M = 200M operations. Might be too slow in Python if not optimized.
    # But we can use bitsets or sets of Fractions.
    
    # Let's try to construct a large set using backtracking with the following pruning:
    # We process elements in increasing order.
    # We maintain `sums` (set of all subset sums of current S).
    # When considering x:
    #   if 1/x in sums: skip x (cannot add because 1/x would be a sum of S, which is S\{x} for the new x).
    #   else: tentatively add x.
    #         new_sums = sums | {s + 1/x for s in sums} | {1/x}
    #         Now we must check if adding x broke the condition for existing elements y.
    #         For each y in S:
    #             if 1/y - 1/x in sums_without_y: 
    #                 # This means 1/y = 1/x + sum(subset of S\{y})
    #                 # So 1/y is a sum of a subset of (S\{y}) U {x}
    #                 # Conflict!
    #                 break
    #         If conflict, backtrack.
    
    # To do this, we need `sums_without_y` for each y in S.
    # We can compute this on the fly or maintain it.
    # Since |S| is small, computing `sums_without_y` by removing y from `sums` is hard because sums are not easily invertible.
    # However, we can recompute `sums_without_y` from scratch for each y? 
    # That would be O(|S| * 2^|S|) per step.
    
    # Let's try a different order: Decreasing order (24 down to 1).
    # Large numbers have small reciprocals.
    # Sums of small reciprocals are small.
    # 1/24, 1/23, ...
    # It's unlikely that a small reciprocal is a sum of other small reciprocals?
    # Actually, 1/24 is very small. The smallest sum of two other reciprocals is 1/23 + 1/22 ~ 0.086. 1/24 ~ 0.041.
    # So 1/24 is smaller than any sum of two or more other reciprocals (from 23..24).
    # So 1/24 can never be a sum of others if we only pick from 24, 23, ...?
    # Wait, if we pick {24, 23, 12}, 1/12 = 1/24 + 1/24? No, distinct elements.
    # 1/12 = 1/24 + 1/24 is not allowed (distinct).
    # 1/12 = 1/24 + 1/x? 1/x = 1/12 - 1/24 = 1/24. x=24. Duplicate.
    # So 1/12 cannot be formed by 1/24 and another distinct element?
    # 1/12 = 1/24 + 1/y => 1/y = 1/24 => y=24.
    # So {12, 24} is valid? 
    # Check: 1/12 not in {1/24}. 1/24 not in {1/12}. Yes.
    
    # Let's try to build a set with large numbers.
    # If we take {13, 14, ..., 24}, is it valid?
    # Current seed returns this. Score 1.0.
    # Can we add 12?
    # Check 1/12. Is it a sum of subset of {13..24}?
    # Min sum of two: 1/24+1/23 ~ 0.086. 1/12 ~ 0.083.
    # So 1/12 < 1/24+1/23.
    # So 1/12 cannot be a sum of two or more elements from {13..24}.
    # Can it be a sum of one? 1/12 = 1/x => x=12. Not in {13..24}.
    # So 1/12 is safe.
    # Now check if adding 12 breaks existing elements.
    # For y in {13..24}, is 1/y a sum of subset of {12} U ({13..24}\{y})?
    # i.e., is 1/y - 1/12 a sum of subset of {13..24}\{y}?
    # 1/y - 1/12 is negative for y>12. 
    # Wait, 1/y < 1/12 for y>12. So 1/y - 1/12 < 0.
    # Sums of reciprocals are positive.
    # So 1/y cannot be 1/12 + (positive sum).
    # So adding 12 to {13..24} is valid!
    
    # So we can include 12.
    # Can we include 11?
    # 1/11 ~ 0.0909.
    # Min sum of two from {12..24}: 1/24+1/23 ~ 0.086.
    # 1/11 > 0.086.
    # So 1/11 MIGHT be a sum of two elements.
    # Check: 1/11 = 1/a + 1/b with a,b in {12..24}, a!=b.
    # 1/a + 1/b = 1/11 => 11(a+b) = ab => ab - 11a - 11b = 0 => (a-11)(b-11) = 121.
    # Factors of 121: 1, 11, 121.
    # a-11=1, b-11=121 => a=12, b=132. (b>24, invalid)
    # a-11=11, b-11=11 => a=22, b=22. (a=b, invalid)
    # So no solution with two distinct elements.
    # What about three elements? 1/11 = 1/a + 1/b + 1/c.
    # This is more likely.
    # 
    # Let's just write the code to find the maximum set using backtracking.
    
    def is_valid_set(S):
        """Check if S is a valid set."""
        if not S:
            return True
        S_set = set(S)
        for x in S:
            others = S_set - {x}
            # Generate all subset sums of others
            # To be efficient, we can stop early if we find 1/x
            target = Fraction(1, x)
            # Use iterative subset sum generation
            # Since |others| <= 23, 2^23 is too big.
            # But we can prune: if sum > target, skip.
            # Reciprocals are positive.
            
            # Let's use a set to store sums.
            sums = set()
            for r in others:
                # Add r to all existing sums
                # We need to be careful to use the state of sums before adding r
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + Fraction(1, r))
                new_sums.add(Fraction(1, r))
                sums = new_sums
                if target in sums:
                    return False
            if target in sums:
                return False
        return True

    # This is_valid_set is O(N * 2^N) which is too slow for N=24 (24 * 16M = 384M).
    # We need a smarter backtracking.
    
    # Let's use the incremental check described earlier.
    # Order: 1 to 24.
    
    best_size = 0
    best_set = []
    
    # We'll use a recursive function.
    # To speed up, we can precompute reciprocals.
    recips = [Fraction(1, i) for i in range(1, 25)]
    
    def backtrack(idx, current_set, sums):
        """
        idx: next index to consider (0-based, corresponding to element idx+1)
        current_set: list of selected elements
        sums: set of all subset sums of reciprocals of current_set
        """
        nonlocal best_size, best_set
        
        # Pruning: if even taking all remaining elements can't beat best_size, return
        if len(current_set) + (24 - idx) <= best_size:
            return
        
        if idx == 24:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Option 1: Skip element idx+1
        backtrack(idx + 1, current_set, sums)
        
        # Option 2: Take element idx+1
        x = idx + 1
        rx = recips[idx]
        
        # Check if rx is in sums
        if rx in sums:
            # Cannot take x because rx would be a sum of a subset of current_set
            # which is S\{x} for the new set.
            pass
        else:
            # Tentatively take x
            # Check if taking x breaks existing elements
            valid = True
            for y in current_set:
                ry = recips[y-1]
                # Check if ry - rx is in sums_without_y
                # We don't have sums_without_y readily available.
                # This is the bottleneck.
                
                # Alternative: Just recompute validity of the new set?
                # No, too slow.
                
                # Let's try a different approach:
                # Since N is small, maybe we can just iterate through all subsets of size k, starting from large k?
                # Or use a SAT solver? No, standard library only.
                
                # Let's try a greedy repair:
                # Start with all 24.
                # While there is a conflict, remove the element that is involved in the most conflicts?
                # Or remove the largest element?
                
                # Let's try a simple local search:
                # 1. Start with S = {1..24}
                # 2. While invalid:
                #    Find a conflict (x, T) where 1/x = sum(1/t for t in T), T subset S\{x}
                #    Remove an element from T or x.
                #    To maximize size, remove the one that causes least damage?
                
                # This is getting complex. Let's go back to backtracking but optimize the check.
                # We can precompute for the current set, for each y, the set of sums of S\{y}.
                # But that's expensive to update.
                
                # Given the time limit, let's try a heuristic:
                # The set {13..24} is valid.
                # We found {12..24} is likely valid.
                # Let's try to construct a large set by adding elements from 1 to 11 to {12..24} and checking.
                
                # Let's just return the best known set for now and refine?
                # No, I need a working algorithm.
                
                # Let's try a different backtracking order: Decreasing.
                # And assume that if we add in decreasing order, the "breaking existing" check is simpler?
                # No.
                
                # Let's try a randomized local search.
                # Start with a random large set.
                # If invalid, remove random conflicting element.
                # If valid, try to add random missing element.
                
                # This is a common approach for such constraints.
                pass
                
            if valid:
                # Update sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + rx)
                new_sums.add(rx)
                
                current_set.append(x)
                backtrack(idx + 1, current_set, new_sums)
                current_set.pop()
                
    # The backtracking above has a flaw: the `valid` check is incomplete because we didn't implement `sums_without_y`.
    # Let's implement a simpler, potentially slower but correct backtracking that recomputes validity for small sets?
    # No.
    
    # Let's try a specific construction.
    # We know {13..24} works.
    # Let's try to add 12, 11, 10, ... one by one and check validity of the whole set using the `is_valid_set` function, but optimized.
    
    # Optimized `is_valid_set`:
    # For each x, check if 1/x is a sum of a subset of S\{x}.
    # We can use a bitset of possible sums? Fractions make this hard.
    # But the denominators are bounded.
    
    # Let's just use the `is_valid_set` function with pruning in the subset sum generation.
    # And use it in a greedy construction.
    
    def greedy_construct():
        S = []
        # Try to add elements in increasing order?
        # Or decreasing?
        # Let's try decreasing: 24, 23, ..., 1
        for x in range(24, 0, -1):
            # Check if S U {x} is valid
            # To check, we only need to check:
            # 1. 1/x not in Sums(S)
            # 2. For y in S: 1/y not in Sums((S\{y}) U {x})
            
            # Let's just call a helper that checks if x can be added to S
            if can_add(x, S):
                S.append(x)
        return S
    
    def can_add(x, S):
        rx = Fraction(1, x)
        # Compute Sums(S)
        sums_S = set()
        for y in S:
            ry = Fraction(1, y)
            new_sums = set(sums_S)
            for s in sums_S:
                new_sums.add(s + ry)
            new_sums.add(ry)
            sums_S = new_sums
            
        if rx in sums_S:
            return False
            
        # Check for each y in S
        for y in S:
            ry = Fraction(1, y)
            # Check if ry is in Sums((S\{y}) U {x})
            # This is equivalent to: ry == rx OR ry - rx in Sums(S\{y})
            # ry == rx implies x=y, impossible.
            # So check if ry - rx in Sums(S\{y})
            
            # Compute Sums(S\{y})
            sums_S_minus_y = set()
            for z in S:
                if z == y:
                    continue
                rz = Fraction(1, z)
                new_sums = set(sums_S_minus_y)
                for s in sums_S_minus_y:
                    new_sums.add(s + rz)
                new_sums.add(rz)
                sums_S_minus_y = new_sums
                
            if ry - rx in sums_S_minus_y:
                return False
                
        return True

    # This can_add is O(|S| * 2^|S|).
    # If |S| is around 15, 2^15 = 32768. 15 * 32768 ~ 500k operations.
    # 24 calls to can_add. 24 * 500k = 12M operations.
    # This should be fast enough in Python? Maybe.
    # Let's try it.
    
    S = greedy_construct()
    
    # Try to improve with local search?
    # For now, return S.
    
    # Ensure non-empty
    if not S:
        return [1]
        
    return S

# EVOLVE-BLOCK-END
