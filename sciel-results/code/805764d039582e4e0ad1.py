import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s budget
    N = 180
    
    # 1. Precompute all forbidden triples (a, b, c) with a < b, c and 1/a = 1/b + 1/c
    # Equation: (b-a)(c-a) = a^2
    # Let d = b-a, e = c-a. d*e = a^2.
    # Since a < b and a < c, d, e >= 1.
    # b = a + d, c = a + e.
    # We need b, c <= N.
    # Also b != c is required for "three distinct elements". 
    # If d == e, then b == c, which is not allowed. So d != e.
    
    # Instead of storing triples, we can store for each 'a', the set of pairs (b,c) that are forbidden.
    # Or better, for each pair (b,c), the set of 'a's that are forbidden.
    # Actually, the condition is: if a in S, then NOT (b in S AND c in S) for any valid (b,c) associated with a.
    # Also, if b in S, then NOT (a in S AND c in S).
    # And if c in S, then NOT (a in S AND b in S).
    
    # Let's precompute the constraints.
    # For each a in [1, N]:
    #   For each divisor d of a^2:
    #     e = a^2 // d
    #     b = a + d
    #     c = a + e
    #     if b <= N and c <= N and b != c:
    #        Add constraint: {a, b, c} cannot be all in S.
        
    # We can represent this as a list of "forbidden pairs" for each element.
    # forbidden_pairs[x] is a set of tuples (y, z) such that if x is in S, 
    # we cannot have both y and z in S.
    
    forbidden_pairs = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Iterate divisors d
        # To avoid duplicates and handle b,c symmetry, we can just iterate all d.
        # But note: (a, b, c) is the same set as (a, c, b).
        # So for a fixed a, if we find d and e, the pair is (a+d, a+e).
        # If we iterate d from 1 to a, we cover all factors.
        # Note: if d == e, b == c, skip.
        for d in range(1, a + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    # The set is {a, b, c}
                    # If a is in S, (b, c) is a forbidden pair.
                    forbidden_pairs[a].add((b, c))
                    # If b is in S, (a, c) is a forbidden pair.
                    forbidden_pairs[b].add((a, c))
                    # If c is in S, (a, b) is a forbidden pair.
                    forbidden_pairs[c].add((a, b))
    
    # Optimization: Convert sets to lists for faster iteration if needed, but sets are good for membership.
    # Actually, checking "if y in S and z in S" is O(1) if S is a set.
    # The cost is iterating over forbidden_pairs[x].
    
    def can_add(S, x):
        """Check if x can be added to S without creating a triple."""
        for (y, z) in forbidden_pairs[x]:
            if y in S and z in S:
                return False
        return True

    def is_valid(S):
        """Check if S contains no triple."""
        for x in S:
            for (y, z) in forbidden_pairs[x]:
                if y in S and z in S:
                    return False
        return True

    # 2. Strategy: Structured Greedy with Backtracking/Repair
    # The key insight is that small numbers create many constraints.
    # Large numbers are "safer" to include because they appear in fewer triples as the smallest element 'a',
    # and even as 'b' or 'c', the required 'a' is small.
    # Let's try a greedy approach that prioritizes including large numbers, 
    # then fills in small numbers if possible.
    
    best_set = set()
    best_size = 0
    
    # We will try multiple orders of addition.
    # Order 1: Descending (Largest to Smallest)
    # Order 2: Ascending (Smallest to Largest) - usually bad because small numbers constrain many others
    # Order 3: Randomized with bias towards larger numbers
    
    def greedy_solve(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    # Try descending order first
    order_desc = list(range(N, 0, -1))
    S_desc = greedy_solve(order_desc)
    best_set = S_desc
    best_size = len(S_desc)
    
    # Try ascending order
    order_asc = list(range(1, N + 1))
    S_asc = greedy_solve(order_asc)
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
    
    # Try randomized orders
    random.seed(42)
    for i in range(100):
        order = list(range(1, N + 1))
        # Bias towards larger numbers? Or just shuffle?
        # Let's just shuffle
        random.shuffle(order)
        S_rand = greedy_solve(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            
    # 3. Local Search / Repair
    # Try to swap elements to increase size.
    # Since the greedy method is "first come first served", it might miss better combinations.
    # We can try to remove an element and add two others? No, size must increase.
    # So we need to find a set S' such that |S'| > |S|.
    
    # A common technique: 
    # 1. Take current best set S.
    # 2. For each element x in S, try removing it.
    # 3. Then greedily add elements from the remaining pool (not in S) in a specific order.
    # 4. If the new set is larger, update.
    
    # Let's implement a "swap" heuristic.
    # Remove x, add y, if valid and size increases (which it won't if 1:1 swap).
    # We need to find a subset of S to remove and a subset of [N]\S to add such that size increases.
    
    # Simpler: 
    # Try to add any element from [N]\S to S. If can't, try removing one element from S to make room for two from [N]\S.
    
    def improve(S, time_limit):
        current = set(S)
        best = set(current)
        best_size = len(current)
        start = time.time()
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try 1-removal, 2-addition
            # This is expensive: O(|S| * |N\S|^2)
            # Let's optimize.
            
            # First, try simple additions (should be none if greedy was maximal, but maybe not)
            for x in range(1, N + 1):
                if x not in current:
                    if can_add(current, x):
                        current.add(x)
                        best = set(current)
                        best_size = len(current)
                        improved = True
                        break
            
            if improved:
                continue
                
            # Try removing x and adding y, z
            # We iterate x in current.
            # For each x, we look for y, z in [N]\current that can be added if x is removed.
            # This is still heavy.
            
            # Alternative: Randomized Local Search
            # Pick a random x in current, remove it.
            # Greedily add elements from [N]\current in random order.
            # If size > best_size, accept.
            
            if len(current) == 0:
                break
                
            x_to_remove = random.choice(list(current))
            trial = current - {x_to_remove}
            
            # Greedily add from remaining
            candidates = [x for x in range(1, N + 1) if x not in trial]
            random.shuffle(candidates)
            
            for y in candidates:
                if can_add(trial, y):
                    trial.add(y)
            
            if len(trial) > best_size:
                best = trial
                best_size = len(trial)
                current = trial
                improved = True
    
        return best

    # Run improvement
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 1.0:
        improved_set = improve(best_set, remaining_time)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)

    # Final validation check (paranoia)
    if not is_valid(best_set):
        # Fallback to a simple valid set
        # All numbers > 90 are likely safe?
        # 1/a = 1/b + 1/c. If a > 90, b,c > 180? No.
        # If a is large, b and c must be larger than a.
        # If we take all numbers in [91, 180], can there be a triple?
        # a in [91, 180]. b > a, c > a.
        # b, c <= 180.
        # So b, c in [a+1, 180].
        # 1/a = 1/b + 1/c.
        # Smallest 1/a is 1/180.
        # Largest 1/b + 1/c is 1/92 + 1/93 approx 0.02.
        # 1/91 approx 0.011.
        # So it's possible.
        # Let's just construct a valid set greedily from scratch if invalid.
        S_fallback = set()
        for x in range(N, 0, -1):
            if can_add(S_fallback, x):
                S_fallback.add(x)
        best_set = S_fallback
        best_size = len(best_set)

    return sorted(best_set)

# EVOLVE-BLOCK-END
