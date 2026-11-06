"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    p = 1009
    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Identify self-inverse residues (x*x % p == 1)
    self_inverse = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inverse.add(x)

    best_start = 1
    best_length = 0

    # For each starting position, find the maximum valid interval length
    for start in range(1, p):  # 1 to 1008
        if time.time() > deadline:
            break

        # Check if start itself is self-inverse
        if start in self_inverse:
            continue

        # Extend the interval from start
        max_end = start
        for end in range(start, p):  # end from start to 1008
            if end in self_inverse:
                break
            # Check if inv[end] is in [start, end-1]
            ie = inv[end]
            if start <= ie <= end - 1:
                break
            max_end = end

        length = max_end - start + 1
        if length > best_length:
            best_length = length
            best_start = start

    if best_length == 0:
        # Fallback: should not happen, but return something valid
        for x in range(1, p):
            if x not in self_inverse:
                return [x, 1]
        return [1, 1]

    return [best_start, best_length]

# EVOLVE-BLOCK-END
