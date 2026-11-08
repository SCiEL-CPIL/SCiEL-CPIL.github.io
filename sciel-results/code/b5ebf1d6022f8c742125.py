"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio V = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is maximized by minimizing the support of h.
    # For non-negative h with max(h)=1 (implied by seed scale), V <= 1/||h||_1.
    # The minimum ||h||_1 is 1 (single spike), giving V=1.
    # The seed (uniform) has ||h||_1=N and V=1/N.
    # Thus, a single spike yields a search score of N/1 = N = 50.
    return [1.0] + [0.0] * (N - 1)
# EVOLVE-BLOCK-END
