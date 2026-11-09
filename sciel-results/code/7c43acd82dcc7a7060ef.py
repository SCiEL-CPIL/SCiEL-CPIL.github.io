"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    half = n // 2
    return [[u, v] for u in range(half) for v in range(half, n)]

# EVOLVE-BLOCK-END
