import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 ≡ 1 (mod 1009) => x = 1 or x = 1008
    self_inverse = set([1, 1008])

    best_start = 1
    best_length = 0

    L = 1
    for R in range(1, N + 1):
        # Check if R itself is self-inverse
        if R in self_inverse:
            # Any window containing R is invalid; push L past R
            L = R + 1
            continue

        # Check if inv[R] is in [L, R-1] (i.e., already in the window)
        if inv[R] >= L and inv[R] < R:
            # inv[R] is in the current window, so we must push L past inv[R]
            L = max(L, inv[R] + 1)

        # Also ensure L is past any self-inverse element at position L
        # (self-inverse elements cannot be in any valid window)
        while L <= R and L in self_inverse:
            L += 1

        # Now [L, R] is valid (or empty if L > R)
        if L <= R:
            current_length = R - L + 1
            if current_length > best_length:
                best_length = current_length
                best_start = L

    if best_length == 0:
        # Fallback: find any valid single element
        for x in range(1, N + 1):
            if x not in self_inverse:
                best_start = x
                best_length = 1
                break

    return [best_start, best_length]
