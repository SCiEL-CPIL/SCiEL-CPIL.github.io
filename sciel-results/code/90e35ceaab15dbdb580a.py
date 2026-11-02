"""Second autocorrelation inequality candidate: use a narrow triangular pulse."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # Nonnegative step heights on 50 equal subintervals of [-1/4, 1/4].
    # A narrow symmetric triangle concentrates mass and keeps the autoconvolution
    # peak at lag 0; this is a valid, stable nonnegative candidate.
    center = N // 2
    h = [0.0] * N
    width = 2  # support radius in samples (so total support = 5 samples)
    for i in range(N):
        d = abs(i - center)
        if d <= width:
            h[i] = float(width - d + 1)
    return h
# EVOLVE-BLOCK-END
