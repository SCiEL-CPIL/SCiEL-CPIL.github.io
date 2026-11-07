def construct():
    A = list(range(17, 33))  # {17,...,32}, sum-free, size 16
    B = list(range(65, 129)) # {65,...,128}, size 64
    return A + B
