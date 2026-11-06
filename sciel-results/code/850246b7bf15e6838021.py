import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    
    def check_add(S, x):
        # Returns True if x can be added to S
        # S is a set
        for b in S:
            den = b - x
            if den == 0: continue
            num = x * b
            if num % den != 0: continue
            c = num // den
            if c in S and c != b and c != x:
                return False
        return True

    def greedy_random():
        items = list(range(1, 181))
        random.shuffle(items)
        S = set()
        for x in items:
            if check_add(S, x):
                S.add(x)
        return S

    def greedy_backward():
        S = set()
        for x in range(180, 0, -1):
            if check_add(S, x):
                S.add(x)
        return S

    def greedy_forward():
        S = set()
        for x in range(1, 181):
            # Forward check logic is different?
            # No, check_add works for any x and S.
            # Because the condition is symmetric in b,c but not a.
            # However, check_add assumes x is the 'a' (smallest)?
            # No, check_add checks if x forms a triple with ANY pair in S.
            # It iterates b in S.
            # If x is the largest, then in the triple (a,b,c), x is b or c.
            # The equation is 1/a = 1/b + 1/c.
            # If x=b, 1/a = 1/x + 1/c => 1/c = 1/a - 1/x.
            # My check_add calculates c = xb / (b-x) assuming x is 'a' and b is 'b'.
            # This only finds triples where x is the SMALLEST element.
            # If x is the LARGEST, this check will MISS the conflict!
            
            # So check_add is only valid if x is the smallest element in any potential triple.
            # This is true if all elements in S are > x.
            # In greedy_backward, this holds.
            # In greedy_forward, S contains elements < x. So x is the largest.
            # So check_add is INCORRECT for greedy_forward.
            
            # We need a check_add_largest(S, x).
            pass

    def check_add_largest(S, x):
        # x is largest. S contains elements < x.
        # x is b or c. Let x=b.
        # 1/a = 1/x + 1/c => 1/c = 1/a - 1/x = (x-a)/(ax)
        # c = ax / (x-a)
        for a in S:
            den = x - a
            if den == 0: continue
            num = a * x
            if num % den != 0: continue
            c = num // den
            if c in S and c != a and c != x:
                return False
        return True

    def greedy_forward_fixed():
        S = set()
        for x in range(1, 181):
            if check_add_largest(S, x):
                S.add(x)
        return S

    best = greedy_backward()
    best_size = len(best)
    
    S2 = greedy_forward_fixed()
    if len(S2) > best_size:
        best = S2
        best_size = len(S2)

    # Random restarts
    count = 0
    while time.time() - start_time < deadline - 5:
        S_r = greedy_random() # Needs to handle both cases?
        # Random order mixes small and large.
        # If x is added, we don't know if it's smallest or largest relative to future elements.
        # But relative to CURRENT S, we can check both.
        
        def check_add_any(S, x):
            # Check if x is smallest
            if not check_add(S, x):
                return False
            # Check if x is largest
            if not check_add_largest(S, x):
                return False
            return True

        # Re-implement greedy_random with check_add_any
        items = list(range(1, 181))
        random.shuffle(items)
        S = set()
        for x in items:
            if check_add_any(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best = S
            best_size = len(S)
        count += 1
        if count > 5000: break # safety

    return list(best)
