import time

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Construct two orthogonal Latin squares of order 10
    # using the structure Z5 x Z2
    
    S1 = [[0]*n for _ in range(n)]
    S2 = [[0]*n for _ in range(n)]
    
    for r in range(n):
        r1 = r // 2
        r2 = r % 2
        for c in range(n):
            c1 = c // 2
            c2 = c % 2
            
            # S1(r,c) = 2(r1+c1) + (r2+c2)
            s1 = (2 * ((r1 + c1) % 5) + (r2 + c2) % 2) % 10
            # Note: (r2+c2)%2 is just (r2+c2) since sum of bits is 0,1,2. 
            # But wait, symbol must be 0..9.
            # 2 * (0..4) is 0,2,4,6,8.
            # (r2+c2) is 0,1,2.
            # Sum is 0..10.
            # If sum is 10, it should be 0?
            # Let's check the range.
            # r1+c1 mod 5 is 0..4. 2*val is 0,2,4,6,8.
            # r2+c2 is 0,1,2.
            # Max sum 8+2=10.
            # If 10, mod 10 is 0.
            # Is this a valid Latin square?
            # We checked earlier that 4c1+c2 covers 0..9.
            # Here the term is 2(r1+c1) + r2+c2.
            # Let's re-verify the Latin property for S1.
            # Fixed r. c varies.
            # c1 varies 0..4, c2 varies 0,1.
            # val = 2(r1+c1) + r2+c2.
            # As c1 changes, 2(r1+c1) changes by 2.
            # Sequence of 2(r1+c1) for c1=0..4 is an arithmetic progression step 2 mod 10.
            # 0,2,4,6,8.
            # Adding r2+c2 (0,1,2) shifts it.
            # Does it cover 0..9?
            # For a fixed r1, r2, the values 2(r1+c1) are 5 distinct even numbers.
            # Adding 0,1,2 to each? No, c2 is coupled to c1? No, c is index.
            # c = 2c1 + c2.
            # c=0 (c1=0,c2=0), c=1 (c1=0,c2=1), c=2 (c1=1,c2=0)...
            # So for each c1, we have two values: base + r2 and base + r2 + 1.
            # base = 2(r1+c1).
            # Values: {base+r2, base+r2+1} for c1=0..4.
            # base takes values 2r1, 2r1+2, ..., 2r1+8.
            # These are 5 disjoint pairs of consecutive integers?
            # 2r1, 2r1+1
            # 2r1+2, 2r1+3
            # ...
            # 2r1+8, 2r1+9
            # Yes, these 10 values are distinct mod 10.
            # So S1 is a Latin square.
            
            S1[r][c] = s1
            
            # S2(r,c) = 2(r1+2c1) + (r2+c2+r1)
            # Let's check if this matches the previous derivation.
            # Previous: S2 = 3r1 + r2 + 4c1 + c2.
            # Here: 2(r1+2c1) + r2+c2+r1 = 2r1+4c1+r2+c2+r1 = 3r1+4c1+r2+c2.
            # Yes, it matches.
            
            s2 = (3 * r1 + 4 * c1 + r2 + c2) % 10
            S2[r][c] = s2
            
    return [S1, S2]
