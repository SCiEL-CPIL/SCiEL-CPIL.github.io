import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    # Two-pointer: find longest valid interval
    # An interval [L, R] is valid iff for all x in [L,R], inv[x] not in [L,R]
    # When extending R, only need to check if inv[R] in [L,R]
    best_len = 0
    best_start = 2
    L = 2  # 1 is self-inverse (1*1%1009==1), so exclude it

    for R in range(2, 1009):
        if time.time() - start_time > 85:
            break
        if L <= inv[R] <= R:
            L = inv[R] + 1
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L

    if best_len == 0:
        return [2, 1]
    return [best_start, best_len]
