"""Rewritten program for distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        # x must be such that 4*x > n, i.e., a = 4*x - n > 0
        # Also, we want x, y, z distinct. We'll enforce x < y < z.
        # From 1/x + 1/y + 1/z = 4/n, with x < y < z:
        # 3/x > 4/n => x < 3n/4
        # 1/x < 4/n => x > n/4
        # So x is in (n/4, 3n/4). Since x is integer, x_min = n//4 + 1.
        # x_max = (3*n - 1) // 4  (since x < 3n/4)
        
        x_max = (3 * n - 1) // 4
        
        # Iterate x from x_min to x_max
        for x in range(x_min, x_max + 1):
            a = 4 * x - n
            b = n * x
            # We need 1/y + 1/z = a/b
            # y > x, and z > y
            # 2/y > a/b => y < 2b/a
            # 1/y < a/b => y > b/a
            # So y is in (b/a, 2b/a).
            # Also y > x.
            y_min = max(x + 1, b // a + 1)
            y_max = 2 * b // a - 1  # strictly less than 2b/a
            
            if y_max < y_min:
                continue
                
            for y in range(y_min, y_max + 1):
                # 1/z = a/b - 1/y = (ay - b) / (by)
                # z = by / (ay - b)
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z <= y:
                    continue
                # Check distinctness: x, y, z are distinct because x < y < z
                # Check denominator bound
                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
                    break
            
            if found is not None:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
