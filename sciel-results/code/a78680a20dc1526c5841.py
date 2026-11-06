import time

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin
    
    S = set()
    
    # Greedy construction from largest to smallest
    # This works well because larger numbers have fewer multiples in the sum range
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        
        conflict = False
        # Check if x divides b+c for any b, c in S with b > x
        # We only need to check pairs from S that are greater than x
        # Because if b <= x, then x is not the smallest 'a' in the tuple (a,b,c)
        # Wait, the condition is for ALL a,b,c in S with a < b <= c.
        # If we add x, x can be 'a', 'b', or 'c'.
        
        # Case 1: x is 'a'.
        # Then b, c must be > x.
        # Check if x | (b+c) for any b,c in S with b > x.
        
        # Case 2: x is 'b' or 'c'.
        # Then a < x.
        # If x is 'b', c >= x.
        # If x is 'c', b <= x.
        # But we are building S. If we add x, and S already contains elements < x?
        # No, we are iterating from 256 down to 1.
        # So S contains only elements > x.
        # Thus, x will be the SMALLEST element in any new triple involving x and elements of S.
        # So x can only play the role of 'a'.
        # It cannot play 'b' or 'c' because there are no elements in S smaller than x to be 'a'.
        # Wait, if S has elements > x, then for any triple (a,b,c) in S ∪ {x}:
        # If x is not in the triple, it was already checked.
        # If x is in the triple:
        #   - If x is 'a', then b,c > x. (b,c from S)
        #   - If x is 'b', then a < x < c or a < x <= c.
        #     But a must be in S ∪ {x}. Since S has no elements < x, a cannot be < x.
        #     So x cannot be 'b' or 'c' in a valid triple where the other elements are from S.
        #     Exception: if we add multiple elements? No, we add one by one.
        #     So when we add x, the only new triples are those where x is the smallest element.
        
        # Therefore, we only need to check if x | (b+c) for b,c in S with b > x.
        
        candidates = [y for y in S if y > x]
        for i in range(len(candidates)):
            for j in range(i, len(candidates)):
                b = candidates[i]
                c = candidates[j]
                if (b + c) % x == 0:
                    conflict = True
                    break
            if conflict:
                break
        
        if not conflict:
            S.add(x)
            
    return list(S)
