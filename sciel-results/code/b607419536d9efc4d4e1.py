import time

def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 80  # safety margin before 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: 4x > n (so a = 4x-n > 0), x >= 1
        # Also x, y, z distinct positive integers
        # Try x from smallest valid upward
        x_min = n // 4 + 1  # smallest x with 4x > n
        # Upper bound on x: we need room for y > x and z > y
        # 1/x + 1/y + 1/z = 4/n, with y > x, z > y
        # 4/n > 3/x (roughly), so x > 3n/4... but that's not tight
        # Actually, 1/x < 4/n - 2/x (since 1/y + 1/z < 2/x)
        # So 4/n - 2/x < 1/x => 4/n < 3/x => x < 3n/4
        # Wait, that gives an upper bound. Let me think more carefully.
        # With y > x and z > y > x: 1/y < 1/x and 1/z < 1/x
        # So 4/n = 1/x + 1/y + 1/z < 3/x => x < 3n/4
        # And 4/n > 1/x => x > n/4
        # So n/4 < x < 3n/4
        x_max = min(3 * n // 4 - 1, MAX_DENOM)
        
        if x_max < x_min:
            answers.append(None)
            continue
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n  # positive
            b = n * x
            
            # y must satisfy: y > x, a*y > b (so z positive), and z > y
            # z = b*y / (a*y - b) > y => b*y > y*(a*y - b) => b > a*y - b => 2b > a*y => y < 2b/a
            # Also z integer: (a*y - b) | b*y
            # Since gcd(a*y - b, b) = gcd(a*y - b, b) ... let me use: d = a*y - b, z = b*y/d
            # d | b*y. Note gcd(d, y) = gcd(a*y - b, y) = gcd(b, y)
            # So we need d/gcd(d,y) | b/gcd(d,y) ... this is getting complex.
            # Just check directly.
            
            y_min = max(x + 1, b // a + 1)
            y_max = min(2 * b // a, MAX_DENOM)
            
            if y_max < y_min:
                continue
            
            # Iterate y
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                
                d = a * y - b  # positive since y > b/a
                if d <= 0:
                    continue
                
                # Check if d divides b*y
                if (b * y) % d != 0:
                    continue
                
                z = b * y // d
                if z <= y or z > MAX_DENOM:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
