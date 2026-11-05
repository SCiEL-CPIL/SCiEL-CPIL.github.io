import time
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (or 90s)

    # LCM of 1 to 24
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # V[i] = L / i
    V = [0] * 25
    for i in range(1, 25):
        V[i] = L // i
        
    # Precompute all subset sums for any subset of indices?
    # 2^24 is 16 million, which is too many to precompute all at once in Python sets quickly enough?
    # 16M integers in a set might be heavy but feasible? 
    # Actually, we can use a bitset or just compute on the fly with memoization?
    # Given the small domain (24 elements), we can represent subsets as bitmasks.
    # Sum of reciprocals for a subset S is sum(V[i] for i in S).
    # We can precompute the sum for every bitmask?
    # 2^24 = 16,777,216. 
    # Storing an array of 16M integers (Python ints are objects, so memory is high ~ 500MB+). 
    # Better to compute sums on the fly or use a dictionary for visited states?
    # Or, since we only need to check specific targets, we can use meet-in-the-middle per check?
    # The previous program did meet-in-the-middle per check.
    
    # Let's try a different approach:
    # The condition is: for all x in S, V[x] is NOT in { sum(V[y] for y in T) | T subset of S\{x}, T non-empty? }
    # Wait, the problem says "sum of reciprocals of a subset of the OTHER selected elements".
    # Does the subset have to be non-empty? "subset" usually includes empty set.
    # If empty set is allowed, sum is 0. V[x] is never 0. So empty set is irrelevant.
    # So we check if V[x] is a subset sum of {V[y] for y in S, y != x}.
    
    # Optimization:
    # If we fix the set S, the check is:
    # For each x in S: V[x] is not a subset sum of V[S\{x}].
    
    # We can precompute the subset sums of the *entire* candidate pool? No, the pool changes.
    
    # Let's stick to the bitmask DP for subset sums but optimize the search.
    # Since N=24, we can use a DP array `dp[mask]` = sum of values in mask?
    # And a set of all possible sums for a given mask?
    
    # Alternative:
    # Iterate through all 2^24 subsets? That's 16M subsets.
    # For each subset, check if it's valid.
    # Checking validity:
    #   For each x in subset:
    #     Check if V[x] is a subset sum of subset \ {x}.
    # This is expensive if done naively for each x.
    
    # However, we can precompute `subset_sum[mask]` = sum of V[i] for i in mask.
    # Then, for a given mask, and a specific x in mask, we want to know if there exists
    # a submask `sub` of `mask ^ (1<<x)` such that `subset_sum[sub] == V[x]`.
    
    # We can precompute a dictionary `sum_to_masks` or just use a set of sums for each mask?
    # That's too much memory.
    
    # Let's use the property that V[i] are large and distinct.
    # Maybe a backtracking search with strong pruning is best.
    # Pruning:
    # If we have a current set S, and we consider adding x.
    # 1. V[x] must not be a subset sum of S.
    # 2. For each y in S, V[y] - V[x] must not be a subset sum of S \ {y}.
    
    # We can speed up the subset sum check using a precomputed table of subset sums for the *current* set S?
    # As S grows, we can maintain the set of all subset sums of S.
    # `sums_S` = set of all subset sums of S.
    # When adding x:
    #   new_sums_S = sums_S U {s + V[x] for s in sums_S}
    #   Check 1: V[x] not in sums_S.
    #   Check 2: For each y in S:
    #             We need to check if V[y] - V[x] is a subset sum of S \ {y}.
    #             We don't have `sums_{S\{y}}` readily available.
    
    # However, note that V[y] - V[x] < V[y].
    # If V[y] - V[x] is a subset sum of S \ {y}, it is also a subset sum of S (since S \ {y} is a subset of S).
    # But the converse is not true.
    # If V[y] - V[x] is a subset sum of S, it might use x.
    # If it uses x, then V[y] - V[x] = sum(T) + V[x] for some T subset of S\{y,x}.
    # Then V[y] = sum(T) + 2*V[x].
    # This is a specific case.
    
    # Let's refine the check for adding x to S:
    # We maintain `sums_S` (set of subset sums of S).
    # To check if x can be added:
    # 1. If V[x] in sums_S: return False.
    # 2. For each y in S:
    #    target = V[y] - V[x]
    #    If target < 0: continue (subset sums are non-negative, and V[y], V[x] > 0)
    #    We need to check if target is a subset sum of S \ {y}.
    #    We can approximate by checking if target is in `sums_S`.
    #    If target is NOT in `sums_S`, it's definitely not in `sums_{S\{y}}`.
    #    If target IS in `sums_S`, it might be formed using x or not.
    #    If it is formed WITHOUT x, then it's a subset sum of S \ {y} (since x is not in S yet, wait, x is being added, so x is NOT in S. S is the current set. x is the new element. So S does not contain x.)
    #    Ah, correct. S is the set of already selected elements. x is a candidate.
    #    So `sums_S` only contains sums using elements of S.
    #    So if target is in `sums_S`, it is a sum of a subset of S.
    #    Does that subset contain y?
    #    If the subset T such that sum(T)=target does NOT contain y, then T is a subset of S\{y}. So it's valid.
    #    If the subset T contains y, then sum(T) = V[y] + sum(T'). Then target = V[y] + sum(T').
    #    But target = V[y] - V[x]. So V[y] - V[x] = V[y] + sum(T') => -V[x] = sum(T'). Impossible since sums are positive.
    #    Wait, V[y] - V[x] could be negative? We checked target < 0.
    #    If target > 0, and target = sum(T) for T subset of S.
    #    Can T contain y?
    #    If y in T, then sum(T) >= V[y].
    #    But target = V[y] - V[x] < V[y].
    #    So T cannot contain y.
    #    Therefore, ANY subset of S that sums to target must be a subset of S \ {y}.
    #    
    #    So the check simplifies to:
    #    For each y in S:
    #       target = V[y] - V[x]
    #       if target > 0 and target in sums_S: return False
    #       (if target == 0, empty set sums to 0, which is in sums_S. Is empty set allowed? Yes. But sum of reciprocals of empty set is 0. 1/y = 0? No. So target=0 is not a problem? 
    #        Wait, the condition is 1/y = sum(1/z for z in T).
    #        If T is empty, sum is 0. 1/y != 0. So target=0 is fine.
    #        But `0` is in `sums_S`. So we should check `target > 0`.)
        
    # This simplification is HUGE. We only need to maintain `sums_S` (a set of integers) and check membership.
    # We do NOT need to recompute subset sums for S\{y}.
    
    # Let's verify this logic again.
    # Condition for validity of S U {x}:
    # For all z in S U {x}, 1/z is not a sum of reciprocals of a subset of (S U {x}) \ {z}.
    
    # Case 1: z = x.
    # 1/x is not a sum of reciprocals of a subset of S.
    # Equivalent to: V[x] not in sums_S.
    
    # Case 2: z = y in S.
    # 1/y is not a sum of reciprocals of a subset of (S \ {y}) U {x}.
    # Let T be a subset of (S \ {y}) U {x}.
    # sum_{t in T} 1/t = 1/y.
    # Multiply by L: sum_{t in T} V[t] = V[y].
    # T can contain x or not.
    # If x not in T, then T is a subset of S \ {y}. sum(T) = V[y].
    # This would mean V[y] is a subset sum of S \ {y}.
    # But S is already a valid set, so V[y] is NOT a subset sum of S \ {y}.
    # So this case is already handled by S being valid.
    # 
    # If x in T, let T = {x} U T', where T' is a subset of S \ {y}.
    # Then V[x] + sum(T') = V[y].
    # sum(T') = V[y] - V[x].
    # So we need to check if V[y] - V[x] is a subset sum of S \ {y}.
    # As derived above, if V[y] - V[x] > 0 and is a subset sum of S, it must be a subset sum of S \ {y} because any subset of S summing to V[y]-V[x] cannot contain y (since V[y]-V[x] < V[y]).
    # So we just check if V[y] - V[x] is in sums_S.
    
    # So the algorithm is:
    # 1. Start with S = [], sums_S = {0}.
    # 2. Iterate candidates x.
    # 3. Check if x can be added:
    #    a. V[x] not in sums_S.
    #    b. For all y in S: V[y] - V[x] not in sums_S (if > 0).
    # 4. If valid, add x to S, update sums_S = sums_S U {s + V[x] for s in sums_S}.
    
    # This is very fast!
    # The size of sums_S can grow to 2^|S|.
    # Max |S| is 24. 2^24 = 16M.
    # A set of 16M integers in Python is large but might fit in memory (approx 100-200MB).
    # The update `sums_S U {s + V[x] for s in sums_S}` takes O(2^k) time.
    # Total time is sum of 2^k for k=0 to |S|-1, which is O(2^|S|).
    # This is feasible for |S| up to ~20-24.
    
    # We want to maximize |S|.
    # We can use a greedy approach with backtracking if greedy fails to find the max.
    # But greedy might not find the optimal set.
    # However, given the structure, a simple greedy might be quite good.
    # Let's try greedy first. If time permits, try to improve with local search or backtracking.
    
    # Greedy:
    # Order candidates. Which order?
    # Smaller reciprocals (larger numbers) are "safer" to add because they are less likely to be formed by sums of others?
    # Or larger reciprocals (smaller numbers) are "harder" to form?
    # Actually, large V[x] (small x) are harder to form as sums of smaller V's?
    # No, V[x] = L/x. Small x -> large V[x].
    # Sum of reciprocals of others: if others are large (small V), their sum is small.
    # So large V[x] is unlikely to be a subset sum.
    # So adding small x (large V[x]) first is safe.
    # Then adding larger x (small V[x]) is risky because V[x] might be formed by sum of previous V's?
    # Wait, if we add small x first, V[x] is large. sums_S contains small numbers. V[x] is not in sums_S.
    # Then we add larger x. V[x] is small. It might be in sums_S.
    # So order: descending x (24, 23, ..., 1) means we add small V's first?
    # No, x=24 -> V[24] = L/24 (small). x=1 -> V[1] = L (large).
    # So descending x adds small V's first.
    # Ascending x adds large V's first.
    
    # Let's try both orders and pick the best.
    
    def greedy_search(order):
        S = []
        sums_S = {0}
        for x in order:
            if V[x] in sums_S:
                continue
            valid = True
            for y in S:
                target = V[y] - V[x]
                if target > 0 and target in sums_S:
                    valid = False
                    break
            if valid:
                S.append(x)
                # Update sums_S
                new_sums = set(sums_S)
                for s in sums_S:
                    new_sums.add(s + V[x])
                sums_S = new_sums
        return S

    # Try different orders
    best_set = []
    
    orders = [
        list(range(24, 0, -1)), # Descending x (small V first)
        list(range(1, 25)),     # Ascending x (large V first)
    ]
    
    for order in orders:
        S = greedy_search(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Try to improve with backtracking?
    # Given the fast check, we can try a limited backtracking search.
    # Or just try more random orders?
    
    # Let's try a few random permutations
    import random
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 25))
        random.shuffle(order)
        S = greedy_search(order)
        if len(S) > len(best_set):
            best_set = S
            
    # If we have time, try to add elements to the best set found?
    # Or try to remove and re-add?
    
    # For now, return the best found.
    if not best_set:
        return [1]
        
    return best_set

# Test locally? No.
