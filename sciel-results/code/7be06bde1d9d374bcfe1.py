import time
import random
from functools import lru_cache

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s, safety margin
    
    # Strategy: Greedy addition from largest to smallest.
    # If we can add x to current set S without creating a subset sum of 200, we add it.
    # This maximizes cardinality by keeping as many large numbers as possible,
    # which are less likely to combine to small targets like 200 compared to small numbers.
    
    A = set()
    
    # We will try to add numbers from 100 down to 1.
    # For each candidate x, we check if adding x to A creates a subset sum of 200.
    # A subset sum of 200 can be formed in two ways:
    # 1. A subset of the existing A sums to 200. (If this was true, we would have rejected A earlier, so this is false).
    # 2. A subset of (A \ {some elements}) sums to 200 - x, and we add x to it.
    #    Wait, it's simpler: We need to check if there exists a subset S' of A such that sum(S') = 200 - x.
    #    If such S' exists, then S' U {x} is a subset of A U {x} summing to 200.
    #    Note: x is not in A yet, so S' must be disjoint from {x}.
    
    # To efficiently check subset sums, we maintain a set of all achievable sums from A.
    # Let reachable_sums be the set of all sums of subsets of A that are <= 200.
    # Initially A is empty, reachable_sums = {0}.
    
    reachable_sums = {0}
    
    # We iterate candidates from 100 down to 1.
    # Why descending?
    # If we pick large numbers first, they are "expensive" and might block many combinations.
    # But if we pick small numbers first, they are "cheap" and flexible.
    # However, the constraint is "no subset sums to 200".
    # Consider the complement: We want to remove as FEW numbers as possible.
    # Or, we want to include as MANY as possible.
    # Greedy inclusion from largest to smallest is a standard heuristic for packing problems.
    # Let's test if this logic holds.
    # If I include 100, I can't include 100 again (distinct elements).
    # If I include 100, I need to avoid subsets summing to 100 from the rest.
    # If I don't include 100, I have more freedom.
    # Usually, including larger items first is better for maximizing count in knapsack-like constraints?
    # Actually, for subset sum avoidance, small numbers are more dangerous because they can form many sums.
    # Large numbers are safer because they overshoot the target quickly or have fewer combinations.
    # Example: {1, 2, ..., 100}.
    # If we keep 100, we must ensure no subset of the rest sums to 100.
    # If we drop 100, we must ensure no subset sums to 200.
    # Dropping 100 relaxes the constraint significantly.
    # So maybe we should DROP large numbers?
    # Wait, the goal is to MAXIMIZE cardinality.
    # So we want to KEEP as many as possible.
    # If keeping 100 forces us to drop many small numbers to avoid sum 100, it might be bad.
    # If dropping 100 allows us to keep all small numbers (if they don't sum to 200), it might be good.
    # Let's check: Sum of 1..19 is 190. 1..20 is 210.
    # So {1, ..., 19} has no subset sum 200?
    # Max sum is 190. So {1, ..., 19} is valid. Size 19.
    # Can we add 20? 1..20. Sum 210.
    # Subset sum 200? 200 - 20 = 180. Is 180 formable by 1..19? Yes.
    # So {1..20} is invalid.
    # What if we drop 1 and keep 2..20?
    # This is getting complex.
    
    # Let's stick to the "Greedy Addition" but decide the order.
    # Hypothesis: Adding numbers in DESCENDING order (100, 99, ...) is better because:
    # If we add 100, we check if 100 is in reachable_sums of current A.
    # If we start with empty A, reachable_sums = {0}.
    # Add 100: Is 200-100=100 in {0}? No. Add 100. A={100}. Reachable={0, 100}.
    # Add 99: Is 200-99=101 in {0, 100}? No. Add 99. A={100, 99}. Reachable={0, 100, 99, 199}.
    # Add 98: Is 200-98=102 in {0, 100, 99, 199}? No. Add 98.
    # ...
    # This seems to allow keeping many large numbers.
    # Let's see how many we can keep.
    # If we keep 100, 99, ..., k.
    # The subset sums will grow.
    # Eventually, we will hit a number x such that 200-x is already reachable.
    
    # Let's try this approach.
    
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if adding x creates a subset sum of 200.
        # This happens if there exists a subset of current A summing to 200 - x.
        # Let target_rem = 200 - x.
        # If target_rem < 0, it's impossible (but x <= 100, so target_rem >= 100).
        # If target_rem == 0, it means x=200, not in range.
        # So we check if target_rem is in reachable_sums.
        
        target_rem = 200 - x
        
        if target_rem in reachable_sums:
            # Adding x would create a forbidden sum. Skip x.
            continue
        else:
            # Safe to add x.
            A.add(x)
            # Update reachable_sums.
            # New sums are old_sums U {s + x for s in old_sums if s + x <= 200}
            new_sums = set()
            for s in reachable_sums:
                new_s = s + x
                if new_s <= 200:
                    new_sums.add(new_s)
            reachable_sums.update(new_sums)
            
    # Ensure non-empty
    if not A:
        # Fallback: return a known valid set, e.g., {100} or {1..19}
        return [100]
        
    return list(A)
