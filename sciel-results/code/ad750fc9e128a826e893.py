import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit

    # Compute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # For interval [s, e] to be valid: for all x in [s, e], inv[x] not in [s, e]
    # For a pair (x, inv[x]) with x != inv[x], let a = min(x, inv[x]), b = max(x, inv[x])
    # If s <= a, then e must be < b to avoid including both
    # So e_max(s) = min{b-1 : a >= s} over all non-self-inverse pairs
    # This is a suffix minimum computation

    # Precompute: for each a in 1..1008, min_b_minus_1[a] = min{b-1 : pair has min=a}
    min_b_minus_1 = [MOD] * (MOD + 1)
    for x in range(1, MOD):
        if x == inv[x]:
            continue  # self-inverse (1 and 1008), skip
        a = min(x, inv[x])
        b = max(x, inv[x])
        if b - 1 < min_b_minus_1[a]:
            min_b_minus_1[a] = b - 1

    # Suffix minimum: suffix_min[s] = min{min_b_minus_1[a] : a >= s}
    suffix_min = [MOD] * (MOD + 1)
    suffix_min[MOD] = MOD
    for s in range(MOD - 1, 0, -1):
        suffix_min[s] = min(min_b_minus_1[s], suffix_min[s + 1])

    # Find best interval
    best_start = 2
    best_len = 0

    for s in range(1, MOD):
        if time.time() > deadline:
            break
        if s == inv[s]:
            continue
        e_max = suffix_min[s]
        if e_max < s:
            continue
        length = e_max - s + 1
        if length > best_len:
            best_len = length
            best_start = s

    # Verify the result
    for x in range(best_start, best_start + best_len):
        for y in range(best_start, best_start + best_len):
            if (x * y) % MOD == 1:
                return [2, 1]

    return [best_start, best_len]
