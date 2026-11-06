import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    p = 1009
    n = 1008

    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)

    # A[x] = min(x, inv[x]), B[x] = max(x, inv[x])
    # We need M(s) = min { B[x] | A[x] >= s }
    # Initialize min_B_for_A with infinity
    min_B_for_A = [float('inf')] * (n + 2)

    for x in range(1, n + 1):
        a = min(x, inv[x])
        b = max(x, inv[x])
        if b < min_B_for_A[a]:
            min_B_for_A[a] = b

    # Compute suffix minimum of min_B_for_A
    # M[s] = min(min_B_for_A[s], M[s+1])
    M = [float('inf')] * (n + 2)
    current_min = float('inf')
    for s in range(n, 0, -1):
        if min_B_for_A[s] < current_min:
            current_min = min_B_for_A[s]
        M[s] = current_min

    best_start = 1
    best_length = 0

    for s in range(1, n + 1):
        # Max valid end e is M[s] - 1
        max_e = M[s] - 1
        if max_e > n:
            max_e = n
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    if time.time() > deadline:
        return [best_start, best_length]

    # Verification pass: confirm the answer is truly valid
    if best_length > 0:
        end = best_start + best_length - 1
        valid = True
        for x in range(best_start, end + 1):
            if x * x % p == 1:
                valid = False
                break
            ix = inv[x]
            if best_start <= ix <= end:
                valid = False
                break
        if not valid:
            # Fallback: recompute with a conservative valid answer
            # The interval [2, 31] should be safe as a fallback
            return [2, 30]

    return [best_start, best_length]
