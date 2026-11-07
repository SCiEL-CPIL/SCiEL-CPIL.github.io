import time

def construct():
    start = time.time()
    deadline = start + 80
    
    N = 128
    all_elems = set(range(1, N+1))
    
    # Start with upper half
    S = set(range(65, N+1))
    
    def is_valid_add(S, x):
        """Check if adding x to S creates a forbidden triple."""
        s = S
        # x could be a, b, or c (one of the base elements)
        # or x could be a+b, a+c, or b+c (one of the sums)
        
        # Case: x is c (the largest base element)
        # Need a < b < x in S, and a+b, a+x, b+x in S
        for a in s:
            if a >= x:
                continue
            for b in s:
                if a >= b or b >= x:
                    continue
                if a+b in s and a+x in s and b+x in s:
                    return False
        
        # Case: x is a (smallest base element)
        # Need a=x, b, c with x < b < c, and x+b, x+c, b+c in S
        for b in s:
            if b <= x:
                continue
            for c in s:
                if c <= b:
                    continue
                if x+b in s and x+c in s and b+c in s:
                    return False
        
        # Case: x is b (middle base element)
        # Need a < x < c, and a+x, a+c, x+c in S
        for a in s:
            if a >= x:
                continue
            for c in s:
                if c <= x:
                    continue
                if a+x in s and a+c in s and x+c in s:
                    return False
        
        # Case: x = a+b for some a < b in S
        # Then need c != a, b in S, and a+c, b+c in S
        for a in s:
            if a >= x:
                continue
            b = x - a
            if b <= a or b not in s:
                continue
            for c in s:
                if c == a or c == b:
                    continue
                if a+c in s and b+c in s:
                    return False
        
        # Case: x = a+c for some a < c in S
        # Then need b != a, c in S, a<b<c, and a+b, b+c in S
        for a in s:
            if a >= x:
                continue
            c = x - a
            if c <= a or c not in s:
                continue
            for b in s:
                if b == a or b == c:
                    continue
                if a < b < c and a+b in s and b+c in s:
                    return False
        
        # Case: x = b+c for some b < c in S
        # Then need a != b, c in S, a<b<c, and a+b, a+c in S
        for b in s:
            if b >= x:
                continue
            c = x - b
            if c <= b or c not in s:
                continue
            for a in s:
                if a == b or a == c:
                    continue
                if a < b < c and a+b in s and a+c in s:
                    return False
        
        return True
    
    # Greedy: try to add elements
    candidates = list(range(1, 65))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if x in S:
                continue
            if is_valid_add(S, x):
                S.add(x)
                improved = True
    
    # Local search: single swaps
    for _ in range(500) and time.time() < deadline:
        progress = False
        for x in list(S):
            if time.time() > deadline:
                break
            for y in all_elems - S:
                S.discard(x)
                if is_valid_add(S, y):
                    S.add(y)
                    progress = True
                    break
                S.add(x)
            if progress:
                break
        if not progress:
            break
    
    return list(S)
