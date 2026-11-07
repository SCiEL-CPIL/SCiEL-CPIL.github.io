import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Initial candidate: Split strategy
    # S_large = {64, ..., 128}, S_small = {1, 3, ..., 31}
    S = set(range(64, 129))
    S.update(range(1, 32, 2))
    
    def is_valid(S):
        # Check if S is valid
        # No three distinct a,b,c in S such that a+b, a+c, b+c are in S
        elements = list(S)
        n = len(elements)
        # Optimization: If min(S) is large, check is fast
        # But we need a general check
        # For each pair (a,b), check if a+b in S
        # If so, check if there is c such that a+c in S and b+c in S
        # This is O(N^3) worst case, but N=128, so 128^3 ~ 2M ops. Fast enough for few calls.
        # But we call it in local search. Need faster check.
        
        # Better check:
        # For each c in S:
        #   For each a in S, a != c:
        #     If a+c in S:
        #       Check if there exists b in S, b != a, b != c, such that b+c in S and a+b in S.
        
        # Even better:
        # Precompute sums?
        
        # Let's use a simple O(N^2) check per candidate move if possible, or O(N^3) for full validation.
        # For local search, we can check validity incrementally.
        
        # For now, full validation.
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                s_ab = a + b
                if s_ab not in S:
                    continue
                # Check for c
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = elements[k]
                    if a + c in S and b + c in S:
                        return False
        return True

    def violation_count(S):
        # Count number of violating triples
        elements = list(S)
        n = len(elements)
        count = 0
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                s_ab = a + b
                if s_ab not in S:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = elements[k]
                    if a + c in S and b + c in S:
                        count += 1
        return count

    # Local Search
    best_S = set(S)
    best_size = len(S)
    
    while time.time() < deadline:
        # Try to add a random element
        if len(S) < 128:
            candidates = [x for x in range(1, 129) if x not in S]
            if candidates:
                x = random.choice(candidates)
                new_S = S | {x}
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    # Try to remove an element to make it valid?
                    # Or just reject.
                    pass
        
        # Try to swap
        if len(S) > 0:
            x = random.choice(list(S))
            candidates = [y for y in range(1, 129) if y not in S]
            if candidates:
                y = random.choice(candidates)
                new_S = (S - {x}) | {y}
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
        
        # Occasionally restart with a different seed
        if random.random() < 0.01:
             # Try a different split
             M = random.randint(64, 80)
             S = set(range(M, 129))
             K = (M-1)//2
             # Pick random sum-free subset of 1..K
             # Odds are a good default
             S.update(range(1, K+1, 2))
             if is_valid(S):
                 if len(S) > best_size:
                     best_S = set(S)
                     best_size = len(S)
             else:
                 # Repair
                 # Remove elements causing violations
                 # Simple repair: remove random element from violation
                 pass

    return list(best_S)
