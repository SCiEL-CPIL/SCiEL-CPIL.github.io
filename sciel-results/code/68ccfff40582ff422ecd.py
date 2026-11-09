import time

def construct():
    answers = [None] * 254
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s wall-clock

    for idx, n in enumerate(range(3, 257)):
        if time.time() - t0 > DEADLINE:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4  # x must be < 3n/4 for 3/x > 4/n to allow y,z positive
        
        # Cap the x range to avoid pathological cases, but allow full range for small n
        # For n up to 256, 3n/4 - n/4 = n/2 <= 128, so the range is at most ~128 values
        for x in range(x_min, x_max + 1):
            if time.time() - t0 > DEADLINE:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # y must satisfy: a*y - b > 0 => y > b/a
            # and z = b*y/(a*y-b) > y => b*y > y*(a*y-b) => b > a*y - b => 2b > a*y => y < 2b/a
            y_lo = b // a + 1
            y_hi = 2 * b // a  # y < 2b/a, so y <= floor(2b/a)
            if y_hi < y_lo:
                continue
            # Also need y > x for distinctness (x < y < z)
            if y_lo <= x:
                y_lo = x + 1
            if y_hi < y_lo:
                continue
            for y in range(y_lo, y_hi + 1):
                if time.time() - t0 > DEADLINE:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y:
                    continue
                if x == y or x == z or y == z:
                    continue
                if x > 1000000000000 or y > 1000000000000 or z > 1000000000000:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        answers[idx] = found

    return answers
