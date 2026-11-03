N = 600

def construct():
    # The goal is to minimize the normalized maximum of the autoconvolution.
    # Assuming the normalization is by the L-infinity norm (max height),
    # the objective is max(h*h) / max(h).
    # This is minimized by minimizing the support of h.
    # A single non-zero interval minimizes the support.
    h = [0.0] * N
    h[0] = 1.0
    return h
