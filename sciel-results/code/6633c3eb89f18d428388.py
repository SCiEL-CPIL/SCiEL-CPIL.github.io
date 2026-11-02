        if g % 2 == 0:
            # even girth
            n = 2 * k
            for i in range(1, g // 2):
                n += 2 * k * (k - 1) ** i
            return n
