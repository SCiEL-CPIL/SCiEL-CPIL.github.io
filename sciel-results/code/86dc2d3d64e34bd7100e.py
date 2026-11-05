"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    N = 1008

    # Precompute modular inverses for 1..1008
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    L = 1
    max_len = 0
    best_L = 1

    for R in range(1, N + 1):
        v = inv[R]
        if v >= L:
            L = v + 1
        
        current_len = R - L + 1
        if current_len > max_len:
            max_len = current_len
            best_L = L

    return [best_L, max_len]
# EVOLVE-BLOCK-END
