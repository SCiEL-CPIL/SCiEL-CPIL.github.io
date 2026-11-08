"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a Sidon set to maximize the ratio.
    # For a Sidon set of size K with values 1/K:
    # ||h||_1 = 1
    # ||h||_2^2 = K * (1/K)^2 = 1/K
    # sup(h*h) = (1/K)^2 (since all sums i+j are distinct)
    # Ratio = (1/K) / (1 * 1/K^2) = K
    # We found a Sidon set of size 7 in [0, 49]: {1, 2, 4, 8, 16, 32, 49}
    h = [0.0] * N
    indices = [1, 2, 4, 8, 16, 32, 49]
    val = 1.0 / len(indices)
    for i in indices:
        h[i] = val
    return h
# EVOLVE-BLOCK-END
