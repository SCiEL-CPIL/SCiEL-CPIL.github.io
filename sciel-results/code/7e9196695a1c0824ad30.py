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

    # For each element i, compute the "conflict bound":
    # - If i is self-inverse (i*i % MOD == 1), it can never be in a valid interval.
    #   We mark it with a special sentinel to force exclusion.
    # - Otherwise, if inv[i] < i, then any interval containing i must start after inv[i].
    #   So the constraint is: start >= inv[i] + 1.
    # - If inv[i] > i, the constraint will be handled when we process inv[i]
    #   (since inv[inv[i]] = i < inv[i], it will impose start >= i + 1).
    #
    # For a forward sweep: for each end e, the valid start s must satisfy
    # s >= max over all x in [s, e] of conflict_bound[x].
    # But this is circular. Instead, use the forward two-pointer which is equivalent.
    #
    # Alternative: backward pass. For each start s, find the maximum valid end e.
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    # For a fixed s, as e increases, the first violation occurs at the smallest e
    # such that inv[e] >= s (and inv[e] <= e, which is automatic if inv[e] >= s > 0).
    # Actually, the violation is: there exists x in [s, e] with inv[x] in [s, e].
    # For fixed s, the first e that causes a problem is the smallest e >= s such that
    # inv[e] >= s. But also, if e itself is self-inverse, it's immediately invalid.
    #
    # Let's use the forward two-pointer (which is correct and O(N)):
    # Maintain s such that [s, e] is valid. When adding e:
    #   1. If e is self-inverse, skip it (s = e + 1).
    #   2. If inv[e] is in [s, e-1], move s to inv[e] + 1.
    #   3. Record length.

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

    # Verify the result is valid
    def is_valid(start, length):
        if start < 1 or start + length - 1 > N or length <= 0:
            return False
        end = start + length - 1
        for x in range(start, end + 1):
            for y in range(x, end + 1):
                if (x * y) % MOD == 1:
                    return False
        return True

    if best_len == 0 or not is_valid(best_start, best_len):
        # Fallback: find any valid single-element interval
        for i in range(2, N):
            if i * i % MOD != 1:
                return [i, 1]
        return [2, 1]

    return [best_start, best_len]
