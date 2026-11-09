import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Base: upper tail [171, 256] is valid
    # For a in [171,256], b,c in [171,256] with a<b<=c:
    # b+c >= 172+172 = 344 > 2*171 = 342, so b+c > 2a
    # b+c <= 512, so m*a = b+c requires m >= 3, but 3a >= 513 > 512 for a>=171
    # Thus no valid m exists, so a never divides b+c
    s = set(range(171, 257))
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without violating the property."""
        # We need: for every a<b<=c in (current_set ∪ {x}), a does NOT divide b+c
        # New violations can only involve x as one of a, b, or c
        # Case 1: x is the smallest (a=x, b,c in current_set with x<b<=c)
        #   Need: x does NOT divide b+c for all b,c in current_set with x<b<=c
        # Case 2: x is the middle (b=x, a in current_set with a<x, c in current_set with x<=c)
        #   Need: a does NOT divide x+c for all a in current_set with a<x, c in current_set with x<=c
        # Case 3: x is the largest (c=x, a,b in current_set with a<b<=x)
        #   Need: a does NOT divide b+x for all a,b in current_set with a<b<=x
        
        elems = sorted(current_set)
        
        # Precompute for efficiency
        # Case 1: x is 'a'
        for b in elems:
            if b <= x:
                continue
            for c in elems:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle)
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest)
        for a in elems:
            for b in elems:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    # Greedily try to add elements from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, s):
            s.add(x)
    
    return sorted(s)
