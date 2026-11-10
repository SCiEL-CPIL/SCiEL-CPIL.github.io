def construct():
    # All numbers are multiples of 6, so all pairwise sums are multiples of 6.
    # Prime factors of any multiple of 6 are subset of {2, 3}.
    # 12 distinct multiples of 6, all <= 10000.
    return [6 * i for i in range(1, 13)]
