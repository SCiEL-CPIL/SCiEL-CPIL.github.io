import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin within 90s limit

    def is_valid(S, x):
        """
        Check if adding x to set S maintains the property:
        For every a < b <= c in the resulting set, a does NOT divide b+c.
        """
        S_list = sorted(S)
        
        # Case 1: x is the smallest element a
        # Need: x < b <= c, and x does not divide b+c
        # Since we're checking against existing S, all elements in S are not necessarily > x.
        # Wait, the greedy approach usually adds one element at a time.
        # The condition must hold for ALL triples in the NEW set.
        # This means:
        # 1. Existing triples in S are already valid.
        # 2. New triples involving x must be valid.
        
        # New triples involving x:
        # (x, b, c) where x < b <= c. b, c in S.
        # (a, x, c) where a < x <= c. a in S, c in S.
        # (a, b, x) where a < b <= x. a in S, b in S.
        
        # Let's check each:
        
        # 1. x < b <= c, b,c in S
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            # b > x
            # c must be >= b and in S
            for j in range(i, len(S_list)):
                c = S_list[j]
                # c >= b is guaranteed by sorted list and j>=i
                # Check x | (b+c)
                if (b + c) % x == 0:
                    return False
        
        # 2. a < x <= c, a in S, c in S
        # This requires a < x and c >= x.
        # And a | (x+c) must be false.
        for a in S_list:
            if a >= x:
                continue
            # a < x
            for c in S_list:
                if c < x:
                    continue
                # c >= x
                # Note: if c == x, then (x,c) is (x,x) but x is not in S yet.
                # Wait, x is the new element. S does NOT contain x yet.
                # So c is from S. If c == x, x is not in S. So c cannot be x.
                # However, the condition is a < x <= c.
                # If x is added, can c be x? No, c is from S.
                # But the triple is (a, x, c). Here b=x, c=c.
                # Condition: a < b <= c => a < x <= c.
                # So c must be >= x. Since c is in S, and x not in S, c > x or c == x?
                # If c is in S, and x not in S, c can be equal to x only if x is already in S? No.
                # So c is strictly different from x unless x was already there, which it isn't.
                # So we just check c >= x.
                if (x + c) % a == 0:
                    return False
        
        # 3. a < b <= x, a in S, b in S
        # Condition: a < b <= x.
        # a | (b+x) must be false.
        for a in S_list:
            for b in S_list:
                if a < b and b <= x:
                    if (b + x) % a == 0:
                        return False
        
        return True

    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid(S, x):
                S.add(x)
        return S

    best_S = set()

    # 1. Descending order (often good for packing)
    S1 = greedy(range(n, 0, -1))
    if len(S1) > len(best_S):
        best_S = S1

    # 2. Ascending order
    if time.time() < deadline:
        S2 = greedy(range(1, n + 1))
        if len(S2) > len(best_S):
            best_S = S2

    # 3. Random shuffles
    if time.time() < deadline:
        for seed in range(10000):
            if time.time() > deadline:
                break
            random.seed(seed)
            order = list(range(1, n + 1))
            random.shuffle(order)
            S3 = greedy(order)
            if len(S3) > len(best_S):
                best_S = S3
                # Optional: local search around best_S? Might be too slow.
                # Just keep greedy for now.

    # 4. Local search / Hill climbing on the best set found?
    # We can try to add elements that were excluded, or swap elements.
    # Given time constraints, simple greedy with many restarts is safer.
    
    # One more strategy: Start with large numbers, then try to fill gaps with smaller ones?
    # The descending greedy already does this.

    if not best_S:
        return [256]
    
    return list(best_S)
