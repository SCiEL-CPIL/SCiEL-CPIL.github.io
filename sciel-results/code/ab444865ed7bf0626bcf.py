"""Combined program for the no-divide-sum-of-three-others task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 192
    
    def can_add(S, x):
        """
        Checks if x can be added to set S without violating the condition.
        Condition: No element in S U {x} divides the sum of any 3 distinct other elements.
        """
        # 1. Check if x divides the sum of any 3 elements in S
        #    If S has < 3 elements, this is false.
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Check if any a in S divides the sum of x and any 2 other elements in S
        #    Sum = x + b + c. We need (x + b + c) % a != 0 for all b, c in S \ {a}, b != c.
        #    This is equivalent to: (b + c) % a != (-x) % a
        for a in S:
            target = (-x) % a
            # We need to check if there exist b, c in S \ {a} such that (b+c) % a == target
            # Get elements in S excluding a
            # Since S is a list, we can just filter
            # Optimization: if a is 1, target is 0. Any b+c is divisible by 1? No, (b+c)%1 is always 0.
            # So if a=1 is in S, we always fail?
            # Wait. If a=1, 1 divides everything. So if 1 is in S, it divides sum of any 3 others.
            # So 1 cannot be in a valid set of size > 3?
            # Actually, if 1 is in S, and we pick 3 others, 1 divides their sum.
            # So 1 is never a valid element if |S| >= 4?
            # Yes. So 1 will be rejected early.
            
            if a == 1:
                # 1 divides any sum. So if there are at least 3 other elements, it's invalid.
                # But we are checking if ADDING x makes it invalid.
                # If a=1 is in S, and we add x, does 1 divide sum of x and 2 others?
                # Yes, 1 divides everything.
                # So if 1 is in S, we can only have 2 other elements?
                # If S has 1 and 2 others, and we add x, S becomes {1, b, c, x}.
                # 1 divides b+c+x. So invalid.
                # So if 1 is in S, we cannot add any x if len(S) >= 2?
                # If len(S) < 2, we can add.
                # But we are checking validity of the NEW set.
                # If a=1 is in S, and we add x, the new set contains 1.
                # The condition for a=1 is: 1 must not divide sum of any 3 distinct others.
                # The others are S \ {1} U {x}.
                # If |S \ {1} U {x}| >= 3, then there exist 3 others, and 1 divides their sum.
                # So if 1 is in the new set, and the new set has size >= 4, it is invalid.
                # So if a=1 is in S, we can only add x if |S| + 1 < 4, i.e., |S| < 3.
                if len(S) >= 3:
                    return False
                # If len(S) < 3, we don't have 3 others yet, so 1 is safe for now?
                # But we must check other constraints.
                # For a=1, the "sum of 3 others" check is the only one for a=1.
                # So if len(S) < 3, a=1 doesn't cause a conflict yet.
                # But we still need to check other a's.
                # So we just continue to the next a.
                continue

            # Filter S to exclude a
            # To speed up, we can precompute S_minus_a? 
            # But S is small.
            S_minus_a = [v for v in S if v != a]
            
            if len(S_minus_a) < 2:
                continue
                
            # Check if any pair in S_minus_a sums to target mod a
            seen = [False] * a
            # Use indices to ensure b != c
            for i in range(len(S_minus_a)):
                b = S_minus_a[i]
                for j in range(i + 1, len(S_minus_a)):
                    c = S_minus_a[j]
                    seen[(b + c) % a] = True
            
            if seen[target]:
                return False
                
        return True

    best_set = []
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 1. Greedy Descending
    S1 = []
    unselected = set(range(1, n + 1))
    for x in range(n, 0, -1):
        if x in unselected:
            if can_add(S1, x):
                S1.append(x)
                unselected.remove(x)
    update_best(S1)

    # 2. Greedy Ascending
    S2 = []
    unselected = set(range(1, n + 1))
    for x in range(1, n + 1):
        if x in unselected:
            if can_add(S2, x):
                S2.append(x)
                unselected.remove(x)
    update_best(S2)

    # 3. Random Greedy
    for _ in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S3 = []
        unselected = set(order)
        for x in order:
            if x in unselected:
                if can_add(S3, x):
                    S3.append(x)
                    unselected.remove(x)
        update_best(S3)

    # 4. Local Search on Best Set
    if best_set:
        current_set = best_set[:]
        unselected = set(range(1, n + 1)) - set(current_set)
        
        attempts = 0
        while time.time() < deadline and attempts < 10000:
            attempts += 1
            if not current_set or not unselected:
                break
            
            # Random Swap
            idx = random.randint(0, len(current_set) - 1)
            a = current_set[idx]
            b = random.choice(list(unselected))
            
            # Check if swap is valid
            S_prime = [x for x in current_set if x != a]
            
            if can_add(S_prime, b):
                current_set[idx] = b
                unselected.remove(b)
                unselected.add(a)
                
                # Try to add more after swap
                # Check if any unselected can be added now
                added = False
                for cand in list(unselected):
                    if can_add(current_set, cand):
                        current_set.append(cand)
                        unselected.remove(cand)
                        update_best(current_set)
                        added = True
                        break
                
    return best_set
# EVOLVE-BLOCK-END
