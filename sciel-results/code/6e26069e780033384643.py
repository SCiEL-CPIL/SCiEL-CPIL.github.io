import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # We want to maximize the size of the subset of [1, 180]
    # such that no three distinct elements a, b, c satisfy 1/a = 1/b + 1/c.
    # This implies a < b and a < c.
    
    S = set()
    
    # Greedy strategy: iterate from largest to smallest.
    # Large numbers are less likely to be the 'a' in a triple with other large numbers
    # because b, c must be larger than a.
    # By including large numbers first, we might block some small numbers,
    # but small numbers block many pairs of large numbers if included.
    # Actually, if we include a small number 'a', it conflicts with pairs (b, c) in S.
    # If we include a large number 'L', it conflicts with pairs (b, c) in S where b, c > L.
    # Since we process in descending order, when we consider 'a', S contains numbers > a.
    # So 'a' conflicts with pairs in S.
    # If we skip 'a', we keep all numbers > a that are compatible.
    # This seems to favor keeping larger numbers.
    
    for a in range(180, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
            
        conflict = False
        # Check if there exist u, v in S such that 1/a = 1/u + 1/v
        # u, v must be distinct and different from a.
        # Since a is the smallest, u, v > a.
        
        # Optimization: 
        # 1/a = 1/u + 1/v => v = ua / (u - a)
        # Iterate u in S
        for u in S:
            if u <= a:
                continue
            # Check if v is an integer
            numerator = u * a
            denominator = u - a
            if numerator % denominator == 0:
                v = numerator // denominator
                # v must be in S and v != u
                if v in S and v != u:
                    conflict = True
                    break
        
        if not conflict:
            S.add(a)
            
    return list(S)
