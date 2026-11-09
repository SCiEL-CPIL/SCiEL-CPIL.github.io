def construct():
        result = []
        for c in range(2):
            for b in range(2):
                for a in range(6):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if val <= 512:
                        result.append(val)
        return result
