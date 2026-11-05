import math

def find_solution(n):
    # x must be > n/4 and <= 3n/4
    x_min = n // 4 + 1
    x_max = 3 * n // 4
    
    for x in range(x_min, x_max + 1):
        A = 4 * x - n
        B = n * x
        
        # 1/y + 1/z = A/B
        # y must be > x
        # y < 2B/A
        # y > B/A
        y_min = max(x + 1, B // A + 1)
        y_max = (2 * B) // A
        
        for y in range(y_min, y_max + 1):
            D = A * y - B
            if D <= 0:
                continue
            if (B * y) % D == 0:
                z = (B * y) // D
                if z > y:
                    # Check distinctness
                    if x != y and y != z and x != z:
                        # Check bound
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            return [x, y, z]
    return None

# Precompute the table
_TABLE = {}
for _n in range(3, 257):
    _TABLE[_n] = find_solution(_n)

def construct():
    return [_TABLE[n] for n in range(3, 257)]
