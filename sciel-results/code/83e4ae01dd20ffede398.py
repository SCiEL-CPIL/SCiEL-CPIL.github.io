"""Find longest interval in [1,1008] with no modular inverse pair (including self)."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s wall-clock limit

    MOD = 1009
    N = 1008

    # Precompute modular inverses: inv[i] = i^(-1) mod MOD
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Two-pointer sweep to find the longest valid interval [s, e]
    # A window [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    # When we extend e to e+1, the new element e+1 conflicts with inv[e+1].
    # We need to ensure inv[e+1] is not in [s, e+1].
    best_start = 1
    best_len = 0
    s = 1

    for e in range(1, N + 1):
        ie = inv[e]
        # Check if ie is in [s, e]
        if s <= ie <= e:
            # Need to move s past ie
            s = ie + 1
        # After adjustment, window is [s, e]
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

        # Check deadline
        if time.time() > deadline:
            break

    if best_len == 0:
        # Should not happen, but fallback
        return [2, 1]

    # Verify the found interval is actually valid
    # For all x, y in [best_start, best_start + best_len - 1], x*y % MOD != 1
    e_end = best_start + best_len - 1
    valid = True
    for x in range(best_start, e_end + 1):
        for y in range(x, e_end + 1):  # symmetric, check x<=y
            if (x * y) % MOD == 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: do a brute-force search for the true longest interval
        # O(N^2 * N) = O(N^3) but N=1008, so ~10^9... too slow.
        # Instead, use the two-pointer but with full verification at each step.
        # Let's redo with a more careful approach.
        best_start = 2
        best_len = 1
        for s_try in range(1, N + 1):
            if time.time() > deadline:
                break
            # Find max valid e for this s
            max_e = s_try - 1
            for e_try in range(s_try, N + 1):
                # Check if [s_try, e_try] is valid
                ok = True
                for x in range(s_try, e_try + 1):
                    ix = inv[x]
                    if s_try <= ix <= e_try:
                        ok = False
                        break
                if ok:
                    max_e = e_try
                else:
                    break
            length = max_e - s_try + 1
            if length > best_len:
                best_len = length
                best_start = s_try

    return [best_start, best_len]
