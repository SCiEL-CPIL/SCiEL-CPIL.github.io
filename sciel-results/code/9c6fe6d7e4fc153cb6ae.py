import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Forward two-pointer sweep: O(N)
    best_len = 0
    best_start = 1

    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break

        # Self-inverse elements cannot be in any valid interval
        if e * e % MOD == 1:
            s = e + 1
            continue

        # If inv[e] is within the current window [s, e], we have a conflict
        ie = inv[e]
        if ie >= s and ie <= e:
            s = ie + 1

        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # Fast verification using inverse lookup (O(n) instead of O(n^2))
    def is_valid_fast(start, length):
        if start < 1 or start + length - 1 > N or length <= 0:
            return False
        end = start + length - 1
        in_interval = set(range(start, end + 1))
        for x in range(start, end + 1):
            if inv[x] in in_interval:
                return False
        return True

    if best_len == 0 or not is_valid_fast(best_start, best_len):
        # Fallback: find any valid single-element interval
        for i in range(2, N):
            if i * i % MOD != 1:
                return [i, 1]
        return [2, 1]

    return [best_start, best_len]
