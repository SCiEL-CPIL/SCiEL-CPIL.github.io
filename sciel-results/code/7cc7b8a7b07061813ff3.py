import time

def construct():
    """
    Find distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns a list of 254 entries, each either None or a list [x, y, z] of distinct
    positive integers <= 10^12 such that 1/x + 1/y + 1/z = 4/n.
    """
    deadline = time.time() + 85.0  # safety margin before 90s kill
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = _find_distinct_triple(n)
        answers[idx] = found
    
    return answers

def _find_distinct_triple(n):
    """
    Find x < y < z such that 1/x + 1/y + 1/z = 4/n.
    Uses the standard transformation:
        1/x + 1/y + 1/z = 4/n
        Let x be the smallest denominator.
        4/n - 1/x = (4x - n)/(nx) = 1/y + 1/z
        Let A = 4x - n, B = nx.
        Then 1/y + 1/z = A/B
        => (y - B/A)(z - B/A) = (B/A)^2  ... but working in integers:
        A/B = 1/y + 1/z = (y+z)/(yz)
        => A*yz = B*(y+z)
        => A*yz - B*y - B*z = 0
        => (Ay - B)(Az - B) = B^2
        Let u = Ay - B, v = Az - B, then u*v = B^2.
        y = (u + B)/A, z = (v + B)/A.
        We need y, z integers, y > x, z > y, and x, y, z distinct.
        
        Since y > x:
            1/y < 1/x
            1/y + 1/z = 4/n - 1/x = A/B
            Since z > y, 1/z < 1/y, so 1/y > (A/B)/2 => y < 2B/A.
            Also 1/y < A/B => y > B/A.
        So B/A < y < 2B/A.
        
        Similarly for z:
            1/z = A/B - 1/y = (Ay - B)/(By)
            z = By/(Ay - B)
            Since z > y:
                By/(Ay - B) > y
                B/(Ay - B) > 1
                B > Ay - B
                2B > Ay
                y < 2B/A  (same bound)
            And z > 0 requires Ay - B > 0 => y > B/A.
        
        So we iterate y from floor(B/A) + 1 to floor(2B/A) - 1 (or similar).
        For each y, compute numerator = B*y, denominator = A*y - B.
        If denominator divides numerator, then z = numerator // denominator.
        Check z > y and distinctness.
    """
    # x must satisfy 1/x < 4/n => x > n/4
    # Also x, y, z distinct positive integers.
    # Smallest x gives largest remaining sum, making it easier to find y, z.
    # x > n/4, so x >= n//4 + 1
    start_x = n // 4 + 1
    
    # Upper bound for x: since y > x, z > y,
    # 4/n = 1/x + 1/y + 1/z < 1/x + 1/(x+1) + 1/(x+2)
    # For large x, this sum approaches 3/x, so 4/n < 3/x => x < 3n/4.
    # But we can be tighter. Actually, since y >= x+1, z >= x+2:
    # 4/n <= 1/x + 1/(x+1) + 1/(x+2)
    # Let's just iterate x from start_x up to some reasonable limit.
    # A safe upper bound: x < 4n/3 (from 1/x + 1/x + 1/x > 4/n => 3/x > 4/n => x < 3n/4... wait)
    # Actually, 1/x + 1/y + 1/z = 4/n with x < y < z implies 3/x > 4/n => x < 3n/4.
    # But also 1/x < 4/n => x > n/4.
    # Let's iterate x from start_x to min(3*n//4, start_x + 1000) to keep it bounded.
    
    # Actually, let's think about the range of x more carefully.
    # We need x < y < z.
    # 4/n = 1/x + 1/y + 1/z < 3/x  =>  x < 3n/4
    # 4/n = 1/x + 1/y + 1/z > 1/x  =>  x < 4n (loose)
    # But also 1/x < 4/n => x > n/4.
    # So x is in (n/4, 3n/4).
    
    max_x = min(3 * n // 4 + 1, start_x + 500)  # Cap search for safety
    
    for x in range(start_x, max_x + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        
        # y range: B/A < y < 2B/A
        # y must be integer > x
        y_low = max(x + 1, B // A + 1)
        y_high = 2 * B // A
        
        # If y_high <= y_low, no valid y in this range
        if y_high <= y_low:
            continue
        
        for y in range(y_low, y_high + 1):
            denom = A * y - B
            if denom <= 0:
                continue
            numer = B * y
            if numer % denom != 0:
                continue
            z = numer // denom
            if z <= y:
                continue
            # Check distinctness: x, y, z are all distinct
            if x != y and y != z and x != z:
                # Check bound
                if z <= 10**12:
                    return [x, y, z]
    
    return None
