import time

def construct():
    deadline = time.time() + 85
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        found = None
        x_min = n // 4 + 1
        for x in range(x_min, x_min + 200000):
            if time.time() > deadline:
                break
            if found:
                break
            y_min = max(x + 1, n // (4 * x) + 1)
            # Search y in a range
            y_search_max = min(y_min + 500000, 2000000)
            for y in range(y_min, y_search_max + 1):
                if time.time() > deadline:
                    break
                denom = 4 * x * y - n
                if denom <= 0:
                    continue
                num = n * (x + y)
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y or z > 1000000000000:
                    continue
                found = [x, y, z]
                break
        answers.append(found)
    return answers
