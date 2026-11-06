import time


def construct():
    p = 1009
    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    best_start = 2
    best_length = 1

    # For each possible start, greedily extend the interval as far as valid
    for start in range(1, p):  # start from 1 to 1008
        S = set()
        length = 0
        for x in range(start, p):  # x from start to 1008
            # Self-inverse check: x*x % p == 1 means x and x form a bad pair
            if (x * x) % p == 1:
                break
            # If the modular inverse of x is already in the interval, invalid
            if inv[x] in S:
                break
            S.add(x)
            length += 1
        if length > best_length:
            best_length = length
            best_start = start

    return [best_start, best_length]
