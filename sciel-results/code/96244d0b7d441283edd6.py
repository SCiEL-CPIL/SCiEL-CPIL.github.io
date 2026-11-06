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
        # Try x values starting from x_min
        for x in range(x_min, x_min + 200):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # Need 1/y + 1/z = a/b
            # => (a*y - b)(a*z - b) = b^2
            # y = (d + b) / a, z = (b*b//d + b) / a for d | b^2
            # Or iterate y: z = b*y / (a*y - b)
            # y must satisfy a*y - b > 0 => y > b/a
            y_min = b // a + 1
            if y_min <= x:
                y_min = x + 1
            # Also need z > y, so a*y - b < b*y/a... let's just iterate
            # z = b*y / (a*y - b), need (a*y - b) | b*y
            for y in range(y_min, y_min + 500):
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Check denominators <= 10^12
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                # Verify: 1/x + 1/y + 1/z = 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
