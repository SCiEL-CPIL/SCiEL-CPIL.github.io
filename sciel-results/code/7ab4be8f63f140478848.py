def construct():
    n = 12
    subset_sums = {0}
    result = []
    for _ in range(n):
        v = 1
        while True:
            new_sums = {s + v for s in subset_sums}
            if not subset_sums.intersection(new_sums):
                subset_sums.update(new_sums)
                result.append(v)
                break
            v += 1
    return result
