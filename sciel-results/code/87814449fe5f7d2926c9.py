"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    N = 1008

    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    best_start = 2
    best_len = 1  # [2,2] is valid: 2*2=4 != 1 mod 1009

    # For each start, find the maximum valid end
    for start in range(1, N + 1):
        # Track which elements are in the current interval [start, end]
        # Condition: no x in interval has inv[x] in interval, and x not self-inverse
        in_interval = [False] * (MOD + 1)
        max_end = start - 1

        for end in range(start, N + 1):
            x = end
            # Diagonal check: x must not be self-inverse (1 or 1008)
            if x == 1 or x == 1008:
                break
            # Check if x's inverse is already in the interval
            if in_interval[inv[x]]:
                break
            # Also check: is x the inverse of some element already in interval?
            # inv[x] in interval means some y in interval has inv[y] = x, i.e., y = inv[x]
            # This is the same check as above due to symmetry of inverse relation.
            in_interval[x] = True
            max_end = end
            length = max_end - start + 1
            if length > best_len:
                best_len = length
                best_start = start

    return [best_start, best_len]
# EVOLVE-BLOCK-END
