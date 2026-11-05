def construct():
    included = set()
    
    for a in range(180, 0, -1):
        can_add = True
        
        # Check all pairs (b, c) in included to see if {a, b, c} forms a forbidden triple
        # We need to check if there exist distinct x, y, z in {a} U included
        # such that 1/x = 1/y + 1/z
        
        # Since we're adding a, the triple must include a.
        # So we check: does there exist b, c in included (b != c) such that
        #   1/a = 1/b + 1/c, OR
        #   1/b = 1/a + 1/c, OR  
        #   1/c = 1/b + 1/a
        # where b, c are distinct elements of included
        
        # Case 1: 1/a = 1/b + 1/c
        for b in included:
            if b == a:
                continue
            # 1/c = 1/a - 1/b = (b - a) / (a * b)
            if b - a == 0:
                continue
            # c = a * b / (b - a)
            num = a * b
            den = b - a
            if den < 0:
                continue  # c would be negative
            if num % den != 0:
                continue
            c = num // den
            if c == a or c == b:
                continue
            if c in included:
                can_add = False
                break
        
        if not can_add:
            continue
        
        # Case 2: 1/b = 1/a + 1/c for some b, c in included
        for b in included:
            if b == a:
                continue
            # 1/c = 1/b - 1/a = (a - b) / (a * b)
            if a - b == 0:
                continue
            num = a * b
            den = a - b
            if den < 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c == a or c == b:
                continue
            if c in included:
                can_add = False
                break
        
        if can_add:
            included.add(a)
    
    return sorted(included)
