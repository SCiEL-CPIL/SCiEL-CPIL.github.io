import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # Also need x >= 1, and we want x < y < z
        # Lower bound: x >= floor(n/4) + 1
        # Upper bound: since 1/x + 1/y + 1/z = 4/n and y > x, z > y
        #   4/n < 3/x => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # inclusive upper bound where 3/x >= 4/n
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n  # positive since x > n/4
            B = n * x
            # We need (y - A)(z - A) = A^2
            # y = A + d, z = A + A^2/d for d | A^2
            # Need y > x, z > y, all distinct, all <= 10^12
            A2 = A * A
            # Find all divisors of A^2
            # Since A <= ~192, A^2 <= ~37000, enumerate divisors
            best = None
            for d in range(1, A2 + 1):
                if A2 % d == 0:
                    e = A2 // d
                    y = A + d
                    z = A + e
                    # Ensure x < y < z and all distinct
                    if x < y < z and y != x and z != y and z != x:
                        if z <= 10**12:
                            best = (x, y, z)
                            break
            if best is not None:
                found = list(best)
                break
        
        answers.append(found)
    
    return answers
