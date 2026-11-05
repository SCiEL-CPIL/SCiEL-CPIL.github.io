import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
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

    def can_add(S_set, S_sorted, x):
        """
        Check if x can be added to S.
        S_sorted is a sorted list of S.
        We need to check three cases:
        1. x is the smallest (a=x). Then b,c in S must have b,c > x.
           Condition: x does not divide b+c for any b,c in S with b,c > x.
        2. x is the middle (b=x). Then a in S must have a < x, and c in S must have c >= x.
           Condition: a does not divide x+c for any a in S with a < x, c in S with c >= x.
        3. x is the largest (c=x). Then a,b in S must have a < b <= x.
           Condition: a does not divide b+x for any a,b in S with a < b <= x.
        """
        if x in S_set:
            return False
        
        m = len(S_sorted)
        
        # Case 1: x is 'a'. x < b <= c.
        # Find indices of elements > x
        # Since S_sorted is sorted, we can binary search or iterate.
        # Iteration is fine for small m.
        greater = [v for v in S_sorted if v > x]
        g_len = len(greater)
        for i in range(g_len):
            b = greater[i]
            if (b + b) % x == 0:
                return False
            for j in range(i + 1, g_len):
                c = greater[j]
                if (b + c) % x == 0:
                    return False

        # Case 2: x is 'b'. a < x <= c.
        # a must be < x. c must be >= x.
        smaller = [v for v in S_sorted if v < x]
        greater_or_equal = [v for v in S_sorted if v >= x]
        
        for a in smaller:
            for c in greater_or_equal:
                if (x + c) % a == 0:
                    return False

        # Case 3: x is 'c'. a < b <= x.
        # a < b <= x. Both a and b are in S.
        # Note: b can be x? No, b is in S, x is new. So b < x.
        # Wait, the condition is a < b <= c. If c=x, then b <= x.
        # Since x is not in S, b must be < x.
        # So we need a, b in S such that a < b < x and a | (b+x).
        for i in range(m):
            b = S_sorted[i]
            if b >= x:
                break
            # a must be < b
            for j in range(i):
                a = S_sorted[j]
                if (b + x) % a == 0:
                    return False

        return True

    def greedy_extend(S, order, time_limit):
        """Greedily add elements to S in given order."""
        S = set(S)
        S_sorted = sorted(S)
        for x in order:
            if time.time() > time_limit:
                break
            if x not in S:
                if can_add(S, S_sorted, x):
                    S.add(x)
                    # Insert x into S_sorted in sorted position
                    # Simple append and sort is O(m log m), but m is small.
                    # Or use bisect.
                    S_sorted.append(x)
                    S_sorted.sort()
        return S

    best_size = 0
    best_S = set()

    # Strategy 1: Start with the known-good upper tail [171, 256] and extend downward
    # This is the strongest seed.
    base_S = set(range(171, n + 1))
    if is_valid_set(base_S):
        best_size = len(base_S)
        best_S = set(base_S)
    
    # Try to extend base_S by adding smaller numbers in descending order
    # Descending order is often better because large numbers are less likely to be 'a' (divisor)
    # and more likely to be 'b' or 'c'. If x is large, it's hard for small a in S to divide x+c or b+x?
    # Actually, if x is large, it's less likely to be divisible by small a? No, x mod a is random.
    # But if x is large, it's less likely to be 'a' for other pairs.
    
    if time.time() < deadline - 5.0:
        # Try descending order from 170 down to 1
        order = list(range(170, 0, -1))
        S_temp = set(base_S)
        S_temp = greedy_extend(S_temp, order, deadline - 2.0)
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)

    # Strategy 2: Randomized restarts with different initial tails and orders
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 123 + 456)
        
        # Vary the starting tail
        tail_start = rng.randint(160, 200)
        S_init = set(range(tail_start, n + 1))
        
        # Verify validity
        if not is_valid_set(S_init):
            # Filter to valid subset
            S_valid = set()
            S_list = sorted(S_init)
            for x in S_list:
                if can_add(S_valid, sorted(S_valid), x):
                    S_valid.add(x)
            S_init = S_valid

        # Generate a random order for the remaining numbers
        remaining = [x for x in range(1, tail_start) if x not in S_init]
        rng.shuffle(remaining)
        
        # Mix: Some descending, some random
        if restart % 2 == 0:
            remaining = sorted(remaining, reverse=True)
        
        S_temp = set(S_init)
        S_temp = greedy_extend(S_temp, remaining, deadline - 2.0)
        
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)

    # Strategy 3: Try to improve best_S by removing an element and adding two others?
    # This is expensive. Instead, try a simple local improvement:
    # Try to add any remaining element to best_S.
    if time.time() < deadline - 2.0 and best_size > 0:
        S_final = set(best_S)
        S_sorted = sorted(S_final)
        # Try all numbers not in S_final
        for x in range(1, n + 1):
            if time.time() > deadline - 1.0:
                break
            if x not in S_final:
                if can_add(S_final, S_sorted, x):
                    S_final.add(x)
                    S_sorted.append(x)
                    S_sorted.sort()
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = set(S_final)

    # Fallback
    if best_size == 0:
        return list(range(129, 257))

    if not is_valid_set(best_S):
        return sorted(set(range(171, 257)))

    return sorted(best_S)
