import time

def construct():
    results = [None] * 254
    MAX_DENOM = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if n % 4 == 0:
            # n = 4k
            # 4/n = 1/k
            # 1/k = 1/(2k) + 1/(3k) + 1/(6k)
            # Denominators: 2k, 3k, 6k
            # n/2, 3n/4, 3n/2
            x = n // 2
            y = 3 * n // 4
            z = 3 * n // 2
            if x != y and y != z and x != z and x > 0 and y > 0 and z > 0 and x <= MAX_DENOM and y <= MAX_DENOM and z <= MAX_DENOM:
                results[idx] = [x, y, z]
            else:
                results[idx] = None
        else:
            x = (n + 3) // 4  # ceil(n/4)
            r = 4 * x - n
            M = n * x
            
            found = None
            # y must be > M/r
            y_start = M // r + 1
            # Limit search range for safety, though it should be found quickly
            for y in range(y_start, y_start + 10000):
                d = r * y - M
                if d <= 0:
                    continue
                if (M * y) % d != 0:
                    continue
                z = (M * y) // d
                if z <= 0:
                    continue
                # Check distinctness
                if x != y and y != z and x != z:
                    # Check bounds
                    if x <= MAX_DENOM and y <= MAX_DENOM and z <= MAX_DENOM:
                        found = [x, y, z]
                        break
            results[idx] = found
            
    return results
