import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    LIMIT = 1009
    MAX_VAL = 1008

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    # Build blocker mask: position i is a blocker if i < inv[i]
    # This means the pair (i, inv[i]) has its smaller endpoint at i
    # An interval [L, R] is valid iff it contains no blocker i where inv[i] is also in [L, R]
    # But actually, the constraint is: for all x,y in [L,R], x*y % 1009 != 1
    # This means: for all x in [L,R], inv[x] must NOT be in [L,R]
    # Equivalently: for all x in [L,R], either inv[x] < L or inv[x] > R

    # The sliding window approach:
    # As we extend R, if inv[R] is in [L, R], we must move L to inv[R]+1
    # This is already optimal in O(N)

    best_len = 0
    best_start = 2

    L = 2
    # R goes from 2 to 1007 (1 and 1008 are self-inverses and cannot be in the interval)
    for R in range(2, 1008):
        # Deadline guard
        if time.time() > deadline:
            break
        # Check if inv[R] is in [L, R]
        if L <= inv[R] <= R:
            L = inv[R] + 1

        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L

    # Ensure we return a valid interval if best_len is 0
    if best_len == 0:
        best_start = 2
        best_len = 1

    return [best_start, best_len]
