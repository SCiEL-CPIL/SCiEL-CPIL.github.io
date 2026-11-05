import math

def construct():
    answers = []
    limit = 1000000000000
    for n in range(3, 257):
        found = None
        # x must be > n/4
        start_x = n // 4 + 1
        # We iterate x up to n, or maybe a bit more?
        # If x is very large, 1/x is small, 1/y+1/z approx 4/n.
        # y,z approx n/2.
        # The range of y is roughly n/2.
        # So x up to n is reasonable.
        for x in range(start_x, n + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            
            # y > nx / A
            y_min = (n * x) // A + 1
            # y <= 2nx / A (assuming y <= z)
            y_max = (2 * n * x) // A
            
            if y_min > y_max:
                continue
                
            # Iterate y
            # Optimization: if y_min is very large, the loop is short?
            # No, y_max - y_min approx nx/A.
            # For x close to n/4, A is small, range is large.
            # For x = n/4 + 1, A ~ 1, range ~ n^2.
            # This could be slow for small x.
            # n=256, x=65. A=4. y_min = 256*65/4 + 1 = 4161.
            # y_max = 2*256*65/4 = 8320.
            # Range size ~ 4000.
            # This is fine.
            
            for y in range(y_min, y_max + 1):
                # z = nxy / ((4x-n)y - nx)
                denom = A * y - n * x
                if denom <= 0:
                    continue
                num = n * x * y
                if num % denom == 0:
                    z = num // denom
                    # Check distinctness
                    if x != y and x != z and y != z:
                        # Check bounds
                        if x <= limit and y <= limit and z <= limit:
                            # Check y <= z is implicit in loop range?
                            # We set y_max based on y<=z assumption.
                            # If y > z, we would have found it when y was smaller?
                            # No, we iterate y. If y > z, then z < y.
                            # But we assumed y <= z to bound the search.
                            # So if we find a solution, it satisfies y <= z.
                            # We just need distinctness.
                            found = [x, y, z]
                            break
            if found:
                break
        answers.append(found)
    return answers
