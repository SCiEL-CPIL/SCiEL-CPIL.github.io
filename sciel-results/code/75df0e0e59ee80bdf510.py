import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    n = 256

    def is_valid_set(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does NOT divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to S."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        
        # Case 1: x is 'a' (smallest)
        # Need x < b <= c in S, and x not divide b+c
        # Since x is new, if x is smallest, all b,c in S are > x.
        # We only need to check pairs in S.
        # Optimization: If x is large, few pairs in S are > x.
        # But if x is small, many pairs.
        
        # Case 2: x is 'b'
        # Need a < x <= c in S. a in S, c in S.
        # Check if a divides x+c.
        
        # Case 3: x is 'c'
        # Need a < b <= x in S. a,b in S.
        # Check if a divides b+x.
        
        # Combined check:
        # For each a in S:
        #   If a < x:
        #     # a can be 'a' in triple (a, b, c) where x is b or c
        #     # If x is 'b': need c in S such that a < x <= c. Check (x+c) % a != 0.
        #     # If x is 'c': need b in S such that a < b <= x. Check (b+x) % a != 0.
        #   If a > x:
        #     # x can be 'a' in triple (x, b, c) where b,c in S.
        #     # Need x < b <= c. Check (b+c) % x != 0.
        
        # Efficient implementation:
        # 1. Check x as 'a':
        #    Iterate pairs (b,c) in S with b > x.
        #    This is expensive if S is large and x is small.
        
        # 2. Check x as 'b' or 'c':
        #    Iterate a in S with a < x.
        #    For each such a, iterate c in S with c >= x (for x as b) and b in S with b <= x (for x as c).
        
        # Let's do a single pass over S to collect constraints.
        # Actually, the previous can_add_fast was O(n^2). Let's stick to that but optimize constants.
        
        S_set = S
        # x as 'a'
        # Find all b in S with b > x
        # For each such b, check b+b and b+c for c > b in S
        # This is O(|{b in S: b>x}|^2)
        
        # x as 'b' or 'c'
        # For each a in S with a < x:
        #   Check if any c in S with c >= x has (x+c)%a==0
        #   Check if any b in S with b <= x has (b+x)%a==0
        
        # To speed up, pre-sort S.
        
        # Simple O(n^2) check is likely fine for n=256, |S|~100.
        # 100^2 = 10,000 checks. 256 * 10,000 = 2.5M. Fast.
        
        for i in range(m):
            a = S_list[i]
            if a < x:
                # x is b or c
                # x is c: need b in S, a < b <= x. (b+x) % a != 0
                for j in range(m):
                    b = S_list[j]
                    if a < b <= x:
                        if (b + x) % a == 0:
                            return False
                # x is b: need c in S, a < x <= c. (x+c) % a != 0
                for j in range(m):
                    c = S_list[j]
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
            elif a > x:
                # x is a
                # Need b,c in S, x < b <= c. (b+c) % x != 0
                # We can break early if we find one.
                # But we need to check all pairs.
                # Optimization: Only check if there are at least two elements > x.
                pass # Handled below to avoid double work or confusion
        
        # x as 'a' check
        # Find indices of elements > x
        greater_indices = [i for i in range(m) if S_list[i] > x]
        for i in range(len(greater_indices)):
            b = S_list[greater_indices[i]]
            if (b + b) % x == 0:
                return False
            for j in range(i + 1, len(greater_indices)):
                c = S_list[greater_indices[j]]
                if (b + c) % x == 0:
                    return False
            
        return True

    best_size = 0
    best_S = set()
    
    # Baseline: Known valid set [171, 256]
    base_S = set(range(171, 257))
    if is_valid_set(base_S):
        best_size = len(base_S)
        best_S = base_S

    # Try to improve by adding elements from [1, 170]
    # Use random restarts with different orderings
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart)
        candidates = list(range(1, 171))
        rng.shuffle(candidates)
        
        S = set(best_S)
        
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if x not in S:
                if can_add(S, x):
                    S.add(x)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            # If we found a larger set, we can try to add more from the remaining pool?
            # The loop above already tried all candidates in that order.
            
        # If we didn't improve, we might try a different base or strategy.
        # But given time, just iterating random orders is a good heuristic.
        
    # Final check
    if not is_valid_set(best_S):
        return sorted(base_S)
        
    return sorted(best_S)
