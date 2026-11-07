"""Optimized program for the suite task; uses a triangular-like shape to maximize L2/(L1*Linfinity)."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    h = [0.0] * N
    # Optimal shape found: h[i] proportional to [1, 1.5, 1]
    # Center at N//2
    c = N // 2
    h[c-1] = 1.0
    h[c] = 1.5
    h[c+1] = 1.0
    return h
# EVOLVE-BLOCK-END
