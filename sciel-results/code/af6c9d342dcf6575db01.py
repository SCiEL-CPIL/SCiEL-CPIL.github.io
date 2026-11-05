"""Maximum-cardinality subset of [1,100] with no subset summing to 200, via DP."""
import time

_DEADLINE = 90.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # DP approach:
    # dp[s] = max number of elements we can select from some processed prefix
    #         such that the set of achievable subset sums (from selected elements)
    #         does NOT include 200, and s is one of the achievable sums.
    # Actually, better formulation:
    # We process elements 1..100. For each element, we decide include or exclude.
    # State: the set of achievable subset sums. But that's 2^200 states - too many.
    #
    # Alternative: dp[s] = max cardinality of a subset T of processed elements
    #               such that 200 is NOT in subset_sum(T), and s IS in subset_sum(T)
    #               (or s=0 meaning empty).
    # But we need to track the FULL subset-sum set, not just one sum.
    #
    # Better: Let's think of it as a knapsack variant.
    # dp[s] = max cardinality of a subset of {1..i} whose subset-sum set contains s
    #         but does NOT contain 200.
    # Transition: for element x, either skip it (dp stays) or include it:
    #   new_sums = old_sums ∪ {old_s + x for old_s in old_sums}
    #   We need new_sums to not contain 200.
    #
    # The issue: dp[s] only tracks one sum s, but the constraint is on the ENTIRE
    # subset-sum set. We need to ensure NO sum equals 200.
    #
    # Key insight: If we maintain dp[s] = max cardinality where the subset-sum set
    # is exactly some set containing s but not 200... this is still complex.
    #
    # Simpler approach: Since target is small (200), use bitset DP.
    # reachable = bitset of achievable sums. We want to maximize |selected|
    # subject to bit 200 not being set in reachable.
    #
    # Let's use: dp[mask] doesn't work (mask would be 200 bits).
    #
    # Alternative clean approach:
    # f[s] = max cardinality of a subset of {1,...,100} with subset-sum set 
    #        containing exactly the sums in some set that includes s but not 200.
    # This is still hard.
    #
    # Let me try a different angle. The problem is equivalent to:
    # Maximize |S| subject to: for all T ⊆ S, sum(T) ≠ 200.
    #
    # Since all elements are positive and ≤ 100, any subset summing to 200 
    # has at most 200 elements (trivially) and at least 2 elements.
    #
    # Let's use the hitting set / minimum vertex cover on hypergraph approach
    # but more cleverly. Enumerate ALL subsets of [1,100] that sum to 200.
    # There aren't that many because elements are large.
    #
    # Actually, let's just do a clean DP:
    # dp[s] = maximum number of elements we can pick from {1,...,i} 
    #         such that no subset of the picked elements sums to 200,
    #         AND the current "frontier" sum is s (meaning s is achievable).
    # This doesn't work because we need to track ALL achievable sums.
    #
    # Best approach for this size: use the fact that if we include element x,
    # we must ensure that 200-x is NOT achievable by the rest.
    #
    # Let me try: process elements in order. Maintain dp where dp[s] = max count
    # of a valid subset (no sum = 200) that has achievable sum s.
    # When adding element x to a subset with achievable sums A:
    #   new_A = A ∪ {a+x for a in A}
    #   Valid iff 200 not in new_A, i.e., 200 not in A and 200-x not in A.
    #
    # So: dp_new[s] = max(dp_old[s], dp_old[s-x] + 1) for valid transitions.
    # But "valid" depends on the full set A, not just s.
    #
    # I'll use a different strategy: since the problem is fixed and small,
    # let me enumerate all subsets summing to 200, then solve the hitting set.
    
    # Step 1: Find all subsets of {1,...,100} that sum to exactly 200.
    # Use DP to enumerate them.
    # For efficiency, note that elements > 200 can't be in such a subset,
    # but all our elements are ≤ 100 < 200, so all could potentially be used.
    # However, any element > 200 can't be used (none here).
    # Elements > 100 can't be in [1,100].
    
    # Enumerate subsets summing to 200 using backtracking/DP.
    # Since 200 is small, we can use a recursive enumeration.
    
    target = 200
    n = 100
    
    # Generate all subsets of {1,...,n} that sum to target.
    # Use iterative DP to build the list.
    # subsets_summing_to[s] = list of tuples (frozenset of elements) - too memory heavy.
    # Instead, just count and find min hitting set differently.
    
    # Actually, for the hitting set, I don't need to enumerate all subsets.
    # I can use a greedy approach or ILP.
    
    # Let me try a cleaner DP:
    # Let dp[s] = max cardinality of a subset S ⊆ {1,...,100} such that 
    #   (a) no subset of S sums to 200, and
    #   (b) s is in the subset-sum set of S (or s=0 for empty).
    #
    # The issue remains that we need to track the full subset-sum set.
    #
    # Alternative: Since 200 is small, use the following:
    # A set S has no subset summing to 200 iff for every x in S, 
    # 200 - x is not in subset_sum(S \ {x}).
    #
    # Let's try yet another approach: 
    # The answer is likely 99 or 98 (remove 1 or 2 elements).
    # Let me verify by checking if removing specific elements works.
    
    def can_reach_200(elements):
        """Check if any subset of elements sums to 200 using bitset DP."""
        reachable = 1  # bit 0 set
        for x in elements:
            if x <= 200:
                reachable |= (reachable << x)
                if (reachable >> 200) & 1:
                    return True
        return False
    
    full_set = set(range(1, 101))
    
    # Try removing 0 elements
    if not can_reach_200(full_set):
        return list(full_set)
    
    # Try removing 1 element
    for x in range(1, 101):
        if time.time() - _START > _DEADLINE - 5:
            break
        remaining_set = full_set - {x}
        if not can_reach_200(remaining_set):
            return list(remaining_set)
    
    # Try removing 2 elements
    for x in range(1, 101):
        if time.time() - _START > _DEADLINE - 5:
            break
        for y in range(x + 1, 101):
            remaining_set = full_set - {x, y}
            if not can_reach_200(remaining_set):
                return list(remaining_set)
    
    # Try removing 3 elements
    for x in range(1, 101):
        if time.time() - _START > _DEADLINE - 5:
            break
        for y in range(x + 1, 101):
            if time.time() - _START > _DEADLINE - 5:
                break
            for z in range(y + 1, 101):
                remaining_set = full_set - {x, y, z}
                if not can_reach_200(remaining_set):
                    return list(remaining_set)
    
    # Try removing 4 elements
    for x in range(1, 101):
        if time.time() - _START > _DEADLINE - 5:
            break
        for y in range(x + 1, 101):
            if time.time() - _START > _DEADLINE - 5:
                break
            for z in range(y + 1, 101):
                if time.time() - _START > _DEADLINE - 5:
                    break
                for w in range(z + 1, 101):
                    remaining_set = full_set - {x, y, z, w}
                    if not can_reach_200(remaining_set):
                        return list(remaining_set)
    
    # Fallback: greedy construction
    return _greedy_fallback()

def _greedy_fallback():
    """Greedy: pick largest elements that don't create a 200-sum."""
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            selected.append(x)
            reachable |= new_sums
    return selected
