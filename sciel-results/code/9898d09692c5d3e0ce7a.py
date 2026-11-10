import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    n = 256
    
    def is_valid(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does not divide b+c"""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(x, S):
        """Check if adding x to S preserves validity. S is a set."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: x is the smallest: x < b <= c, check x does not divide b+c for b,c in S with b > x
        # Find all elements in S greater than x
        greater = [v for v in S_list if v > x]
        for i in range(len(greater)):
            b = greater[i]
            for k in range(i, len(greater)):
                c = greater[k]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle: a < x <= c, check a does not divide x+c for a in S with a < x, c in S with c >= x
        less = [v for v in S_list if v < x]
        greater_or_equal = [v for v in S_list if v >= x]
        for a in less:
            for c in greater_or_equal:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest: a < b <= x, check a does not divide b+x for a,b in S with a < b <= x
        less_or_equal = [v for v in S_list if v <= x]
        for i in range(len(less_or_equal)):
            a = less_or_equal[i]
            for j in range(i+1, len(less_or_equal)):
                b = less_or_equal[j]
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def can_remove(x, S):
        """Check if removing x from S preserves validity (always true since removing can't create violations)"""
        return True
    
    def greedy(order):
        """Greedy construction with given order of elements to try"""
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        return S
    
    # Multiple greedy strategies with different orderings
    best_S = set()
    
    # Strategy 1: Descending order (largest first)
    S1 = greedy(range(256, 0, -1))
    if len(S1) > len(best_S):
        best_S = set(S1)
    
    # Strategy 2: Ascending order (smallest first)
    S2 = greedy(range(1, 257))
    if len(S2) > len(best_S):
        best_S = set(S2)
    
    # Strategy 3: Try only numbers > 128 first, then fill in
    order3 = list(range(256, 128, -1)) + list(range(128, 0, -1))
    S3 = greedy(order3)
    if len(S3) > len(best_S):
        best_S = set(S3)
    
    # Strategy 4: Randomized orderings
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        S_rand = greedy(order)
        if len(S_rand) > len(best_S):
            best_S = set(S_rand)
    
    # Local search: try to add elements that were rejected
    if time.time() < deadline - 10:
        current = set(best_S)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in range(1, 257):
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    improved = True
        if len(current) > len(best_S):
            best_S = set(current)
    
    # Local search: swap elements - remove a small element, try to add larger ones
    if time.time() < deadline - 10:
        current = set(best_S)
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            # Try removing a small element and adding a larger one
            smalls = sorted([x for x in current if x < 128])
            if not smalls:
                break
            removed = random.choice(smalls)
            new_current = set(current)
            new_current.discard(removed)
            added = False
            for x in range(256, removed, -1):
                if x not in new_current and can_add(x, new_current):
                    new_current.add(x)
                    added = True
                    break
            if added and len(new_current) > len(current):
                current = new_current
                if len(current) > len(best_S):
                    best_S = set(current)
    
    # More aggressive local search: try multiple swaps
    if time.time() < deadline - 5:
        current = set(best_S)
        for _ in range(300):
            if time.time() > deadline - 5:
                break
            # Pick a random element to remove
            if not current:
                break
            removed = random.choice(list(current))
            new_current = set(current)
            new_current.discard(removed)
            
            # Try to add as many new elements as possible
            added_any = False
            for x in range(256, 0, -1):
                if x not in new_current and can_add(x, new_current):
                    new_current.add(x)
                    added_any = True
            
            if len(new_current) > len(current):
                current = new_current
                if len(current) > len(best_S):
                    best_S = set(current)
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 2:
        current = set(best_S)
        for x in range(1, 257):
            if x not in current and can_add(x, current):
                current.add(x)
        if len(current) > len(best_S):
            best_S = set(current)
    
    if not best_S:
        return [256]
    
    return list(best_S)
