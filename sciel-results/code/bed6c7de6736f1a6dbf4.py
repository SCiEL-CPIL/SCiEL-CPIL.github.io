"""Construct 12 distinct positive integers with all 4096 subset sums distinct.
The theoretical lower bound for the largest element is 2^(n-1). Powers of 2 achieve it.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576
    such that all 2^12 = 4096 subset sums are distinct.
    Minimize the largest integer.
    
    Uses a greedy algorithm to select the smallest possible next integer
    that preserves the distinctness of subset sums. This is expected to
    yield the powers of two, which is the provably optimal solution.
    """
    n = 12
    max_val = 1048576
    deadline = time.time() + 40

    selected = []
    # Initially, the only subset sum is 0 (empty set)
    current_sums = {0}
    
    for i in range(n):
        # We want to find the smallest integer x > 0 such that
        # adding x to the set of chosen numbers keeps all subset sums distinct.
        # The new sums will be {s, s+x for s in current_sums}.
        # We need {s for s in current_sums} and {s+x for s in current_sums} to be disjoint.
        # This is equivalent to x not being in {s1 - s2 for s1, s2 in current_sums}.
        # However, since we build up, a simpler check is:
        # new_sums = current_sums | {s + x for s in current_sums}
        # len(new_sums) must be 2 * len(current_sums)
        
        # Also x must be distinct from previously selected numbers?
        # The problem states "12 distinct positive integers".
        # If x is already in selected, it's not distinct.
        # But the subset sum distinctness condition is stronger.
        # If x is in selected, say x = y_k, then subset {y_k} and subset {x} (if x was new) ... wait.
        # If we pick x, it becomes part of the set.
        # The condition "distinct subset sums" implies all elements must be distinct?
        # If a = b, then {a} and {b} have same sum. So yes, elements must be distinct.
        
        # Greedy step:
        # Try x = 1, 2, 3, ...
        # Actually, the smallest possible x is 1.
        # Then 2.
        # Then 4.
        # The difference set D = {s1 - s2 | s1, s2 in current_sums} contains all possible "conflicts".
        # x must not be in D (except 0, but x>0).
        # Note that 0 is always in D (s-s). x must be positive.
        
        # Optimization: The smallest valid x is usually small.
        # We can just iterate x upwards.
        
        # To ensure we don't loop forever, we set a reasonable upper bound for x in this step.
        # Since we expect powers of 2, x will be 2^i.
        # Max x is 2^11 = 2048.
        
        found = False
        # Start from 1. We can also start from max(selected)+1 to ensure distinctness faster?
        # No, we need distinct subset sums.
        # Let's just iterate x.
        
        # Efficient check:
        # If we have current_sums, the new sums are current_sums and current_sums + x.
        # They intersect iff there exist s1, s2 in current_sums such that s1 = s2 + x => x = s1 - s2.
        # So x must not be in the set of differences.
        
        # For small n, calculating differences is O(K^2) where K is number of sums.
        # K starts at 1, doubles each step.
        # Step 1: K=1. D={0}. x=1.
        # Step 2: K=2. D={0, 1, -1}. x=2.
        # Step 3: K=4. D={0, 1, 2, 3, -1, -2, -3}. x=4.
        # This is fast enough for n=12.
        
        # We will iterate x from 1 upwards.
        # We know x must be > 0.
        # We also know x should ideally be large enough to avoid collisions.
        
        # Let's just loop x.
        # Upper bound for x in this iteration: max_val.
        
        # To speed up, we can note that if we have picked k elements, the smallest next element
        # in a superincreasing sequence (which powers of 2 are) is sum(previous) + 1?
        # No, powers of 2: 1, 2, 4. Sum prev = 1. Next = 2. 1+1=2.
        # 1, 2, 4. Sum prev = 3. Next = 4. 3+1=4.
        # 1, 2, 4, 8. Sum prev = 7. Next = 8. 7+1=8.
        # So x = sum(selected) + 1 is a candidate.
        # Is it always the smallest?
        # Let S be the set of sums.
        # Max sum is M.
        # Differences are in [-M, M].
        # If x <= M, there might be a collision.
        # Specifically, if x is in the difference set, collision.
        # The difference set contains all integers from -M to M?
        # For powers of 2, sums are 0..2^k-1. Differences are -(2^k-1)..2^k-1.
        # So any x <= 2^k-1 is a difference.
        # Thus x must be > 2^k-1.
        # Smallest integer > 2^k-1 is 2^k.
        # Which is sum(selected) + 1.
        
        # So we can just set x = sum(selected) + 1.
        # Let's verify this logic.
        # If we pick x = sum(selected) + 1, is it always valid?
        # Yes, because all previous sums are in [0, sum(selected)].
        # New sums are in [0, sum(selected)] and [x, x + sum(selected)].
        # Since x > sum(selected), the intervals are disjoint.
        # Thus, all new sums are distinct from old sums.
        # And old sums are distinct by induction.
        # So this construction always works.
        # And it minimizes x at each step?
        # Yes, because any x <= sum(selected) is a difference between two sums?
        # Wait. Are ALL integers in [1, sum(selected)] differences?
        # If the set of sums is "complete" (contains all integers from 0 to S), then yes.
        # Powers of 2 produce complete sets of sums [0, 2^k-1].
        # Does the greedy choice of smallest valid x maintain completeness?
        # If we pick x = S+1, new sums are [0, S] and [S+1, 2S+1].
        # Union is [0, 2S+1]. This is complete.
        # So the property holds.
        # Thus, the greedy algorithm of picking x = sum(selected) + 1 yields 1, 2, 4, 8...
        
        # Let's implement this simple logic.
        
        if not selected:
            x = 1
        else:
            x = sum(selected) + 1
            
        # Check if x is valid (distinct sums)
        # And x <= max_val
        if x > max_val:
            # This should not happen for n=12, 2^11 = 2048 < 1048576
            # But just in case, we might need to search for a larger x?
            # No, if sum+1 > max_val, we are stuck?
            # Actually, we just need *a* valid x.
            # If sum+1 is too big, maybe a bigger x works? No, bigger x is even bigger.
            # So if sum+1 > max_val, no solution with this prefix?
            # But we are minimizing the largest element.
            # For n=12, the optimal max is 2048.
            # So we should be fine.
            
            pass

        # Verify distinctness just to be safe (O(2^k))
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + x)
        
        if len(new_sums) != 2 * len(current_sums):
            # This should not happen with x = sum+1
            # But if it does, we need to find a valid x.
            # Let's implement a fallback search if the simple rule fails.
            # Though it shouldn't.
            found_x = False
            for candidate in range(1, max_val + 1):
                if candidate in selected:
                    continue
                test_sums = set(current_sums)
                for s in current_sums:
                    test_sums.add(s + candidate)
                if len(test_sums) == 2 * len(current_sums):
                    x = candidate
                    found_x = True
                    break
            if not found_x:
                raise ValueError("No valid integer found")
        
        selected.append(x)
        current_sums = new_sums

    # Final verification
    if len(set(selected)) != n:
        raise ValueError("Values not distinct")
    if any(x <= 0 for x in selected):
        raise ValueError("Values must be positive")
    if max(selected) > max_val:
        raise ValueError("Value exceeds maximum")
        
    # Check all subset sums
    all_sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += selected[i]
        all_sums.add(s)
    if len(all_sums) != (1 << n):
        raise ValueError("Subset sums not distinct")

    return selected
# EVOLVE-BLOCK-END
