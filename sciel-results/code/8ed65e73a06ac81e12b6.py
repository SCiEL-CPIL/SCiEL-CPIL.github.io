import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    def is_valid_triple(a, b, c):
        """Check if a,b,c are distinct and form a reciprocal triple in some order."""
        if a == b or b == c or a == c:
            return False
        # 1/a = 1/b + 1/c  <=>  bc = a(b+c)
        if b * c == a * (b + c):
            return True
        # 1/b = 1/a + 1/c  <=>  ac = b(a+c)
        if a * c == b * (a + c):
            return True
        # 1/c = 1/a + 1/b  <=>  ab = c(a+b)
        if a * b == c * (a + b):
            return True
        return False
    
    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        S_list = list(S)
        n = len(S_list)
        # Case 1: x is the 'a' in 1/x = 1/b + 1/c, so b*c = x*(b+c)
        for i in range(n):
            for j in range(i + 1, n):
                b, c = S_list[i], S_list[j]
                if b * c == x * (b + c):
                    return True
        # Case 2: some b in S is the 'a', x and c are the 'b','c'
        # 1/b = 1/x + 1/c => x*c = b*(x+c) => c = b*x/(x+b) must be in S
        # Case 3: some c in S is the 'a', x and b are the 'b','c'
        # By symmetry, same check
        for b in S_list:
            # Check if x and some c in S satisfy 1/b = 1/x + 1/c
            # => x*c = b*(x+c) => c*(x-b) = b*x => c = b*x/(x-b) if x != b
            if x != b:
                num = b * x
                den = x - b
                if den != 0 and num % den == 0:
                    c = num // den
                    if c > 0 and c <= 180 and c != x and c != b and c in S:
                        return True
            # Also check 1/x = 1/b + 1/c form where b is one part
            # 1/x = 1/b + 1/c => b*c = x*(b+c) => c = x*b/(x-b) if x != b
            if x != b:
                num = x * b
                den = x - b
                if den != 0 and num % den == 0:
                    c = num // den
                    if c > 0 and c <= 180 and c != x and c != b and c in S:
                        return True
        return False
    
    # Greedy construction
    S = set(range(91, 181))
    
    # Try adding elements from 1 to 90
    # Order: try smaller numbers first (they're more restrictive)
    candidates = list(range(1, 91))
    random.shuffle(candidates)
    
    for x in candidates:
        if x in S:
            continue
        if not check_conflict(x, S):
            S.add(x)
    
    best_size = len(S)
    best_S = set(S)
    
    # Local search: try swaps and additions
    remaining = [x for x in range(1, 181) if x not in S]
    
    # Try to add more elements
    for x in remaining:
        if not check_conflict(x, S):
            S.add(x)
            best_size = len(S)
            best_S = set(S)
    
    # Local search: remove one, try to add two
    while time.time() < deadline:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for remove_elem in S_list:
            if time.time() > deadline:
                break
            S.remove(remove_elem)
            
            # Try to add elements that were previously blocked
            added = []
            for x in remaining:
                if x in S:
                    continue
                if not check_conflict(x, S):
                    S.add(x)
                    added.append(x)
            
            if len(added) > 1:
                best_size = len(S)
                best_S = set(S)
                improved = True
                # Update remaining
                remaining = [x for x in range(1, 181) if x not in S]
            else:
                # Revert
                for x in added:
                    S.discard(x)
                S.add(remove_elem)
        
        if not improved:
            # Try single additions again
            for x in remaining:
                if not check_conflict(x, S):
                    S.add(x)
                    remaining.remove(x)
                    best_size = len(S)
                    best_S = set(S)
                    break
            else:
                break
    
    # Final check
    result = list(best_S)
    if not result:
        result = list(range(91, 181))
    
    return result
