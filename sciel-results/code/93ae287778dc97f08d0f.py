            available_sorted = sorted(available, key=lambda x: (x[1], x[0]))
            result = [None] * 101
            for r, s, t in available_sorted:
                if result[r] is None:
                    result[r] = t
