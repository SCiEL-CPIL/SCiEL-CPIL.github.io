def construct():
    n = 512
    # Generate all numbers with base 4 digits in {0, 1, 2}
    # Max 5 digits: 4^5 = 1024 > 512
    # We iterate through all combinations of 5 digits (0,1,2)
    # and check if the value is <= 512
    
    S = set()
    # 5 digits: d4 d3 d2 d1 d0
    for d4 in range(3):
        for d3 in range(3):
            for d2 in range(3):
                for d1 in range(3):
                    for d0 in range(3):
                        val = d4 * 4**4 + d3 * 4**3 + d2 * 4**2 + d1 * 4 + d0
                        if 1 <= val <= n:
                            S.add(val)
    
    # If S is empty (unlikely), fallback to a simple set
    if not S:
        return [1]
        
    return sorted(S)
