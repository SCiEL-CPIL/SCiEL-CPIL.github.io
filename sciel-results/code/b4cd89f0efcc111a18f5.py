"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # The task: find a nonempty subset S of [1,100] with maximum cardinality
    # such that no subset of S sums to exactly 200.
    
    # Analytical Insight:
    # 1. No two distinct elements in [1, 100] sum to 200 (max pair sum is 100+99=199).
    # 2. Therefore, any forbidden subset must have size >= 3.
    # 3. Consider the set S = {66, 67, ..., 100}.
    #    - Size: 35.
    #    - Min sum of 3 distinct elements: 66 + 67 + 68 = 201.
    #    - Since all elements are positive, any subset of size >= 3 has sum >= 201.
    #    - Any subset of size 1 has sum <= 100.
    #    - Any subset of size 2 has sum <= 199.
    #    - Thus, no subset sums to 200.
    # 4. Can we do better than 35?
    #    - Try adding 65 to S: {65, ..., 100} (size 36).
    #      - 65 + 66 + 69 = 200. Invalid.
    #    - Try other sets of size 36.
    #      - It is generally difficult to find a set of size 36 avoiding sum 200.
    #      - The "density" of the set [1, 100] is high enough that any 36 elements
    #        likely contain a subset summing to 200.
    #      - For instance, if we take {1, ..., 35, 100}, we can form 100 from {1..35}
    #        and add the 100 element? No, 100 is in the set. We need subset of rest to sum 100.
    #        {1..35} can form 100. So {1..35, 100} is invalid.
    #    - Based on the interval structure, 35 is the optimal maximum cardinality.
    
    # Candidate: {66, ..., 100}
    candidate = list(range(66, 101))
    
    # Verification function
    def is_valid(s):
        # s is a list of integers
        # Check if any subset sums to 200
        reachable = {0}
        for x in s:
            # New sums are formed by adding x to all previously reachable sums
            # We only care about sums <= 200
            new_sums = {y + x for y in reachable if y + x <= 200}
            if 200 in new_sums:
                return False
            reachable |= new_sums
        return True

    # Verify the analytical solution
    if is_valid(candidate):
        return candidate

    # Fallback: Greedy construction
    # This is extremely unlikely to be triggered, but ensures validity if the analytical
    # insight has a flaw (e.g. if I miscounted the min sum).
    # Greedy: Add largest elements first.
    selected = []
    reachable = {0}
    
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if adding x creates a sum of 200
        # This happens if (200 - x) is in reachable
        if (200 - x) not in reachable:
            selected.append(x)
            # Update reachable sums
            new_sums = {y + x for y in reachable if y + x <= 200}
            reachable |= new_sums
        else:
            # Cannot add x
            pass
    
    # If greedy produced an empty set (should not happen), return a valid singleton
    if not selected:
        return [1]
        
    return selected
# EVOLVE-BLOCK-END
