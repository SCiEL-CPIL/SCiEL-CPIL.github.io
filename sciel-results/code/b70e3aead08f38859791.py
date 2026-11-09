import time

def construct():
    answers = []
    deadline = time.time() + 85  # Safety margin
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        # x is the smallest denominator
        # 1/x < 4/n => x > n/4
        # 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue

            # 1/y + 1/z = a/b
            # z = by / (ay - b)
            # Need ay - b > 0 => y > b/a
            # Need z > y => by / (ay - b) > y => b > ay - b => ay < 2b => y < 2b/a
            # Need y > x (distinctness and ordering)
            
            y_low = max(x + 1, b // a + 1)
            y_high = (2 * b) // a
            
            if y_low > y_high:
                continue

            for y in range(y_low, y_high + 1):
                if time.time() > deadline:
                    break
                
                denom = a * y - b
                if denom <= 0:
                    continue
                
                if (b * y) % denom != 0:
                    continue
                
                z = (b * y) // denom
                
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Verify
                # 1/x + 1/y + 1/z = 4/n
                # n(yz + xz + xy) = 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break

        answers.append(found)

    return answers
