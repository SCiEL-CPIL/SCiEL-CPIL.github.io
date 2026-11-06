import time

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        # x must satisfy 4x > n, i.e., x >= n//4 + 1
        x_min = n // 4 + 1
        # Upper bound on x: since y > x and z > y, 1/x + 1/y + 1/z < 3/x, so 3/x > 4/n -> x < 3n/4
        # Also 4/n = 1/x + 1/y + 1/z > 1/x + 1/x + 1/x = 3/x? No, that's wrong direction.
        # 1/x + 1/y + 1/z with x<y<z, the sum is between 1/z+1/z+1/z=3/z and 1/x+1/x+1/x=3/x.
        # We need sum = 4/n. Since x is smallest, 3/x > 4/n -> x < 3n/4.
        # So x in [n//4+1, 3n//4 - 1] roughly.
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # y range: floor(b/a)+1 to floor(2*b/a)
            y_lo = b // a + 1
            y_hi = 2 * b // a
            if y_hi < y_lo:
                continue
            # Also y must be > x
            if y_lo <= x:
                y_lo = x + 1
            if y_hi < y_lo:
                continue
            # Cap the search range to avoid too-long loops for hard cases
            # But we want to find solutions, so let's try
            for y in range(y_lo, y_hi + 1):
                if time.time() > deadline:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y:
                        found = [x, y, z]
                        break
            if found:
                break
        answers[idx] = found
    
    return answers
