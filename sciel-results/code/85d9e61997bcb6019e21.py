"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Strategy: Use the 12 smallest positive zeros of J_{1/2}(x) = sin(sqrt(x))/sqrt(x),
    which are x_i = (i*pi)^2 for i=1..12. These exploit asymptotic orthogonality
    of Laguerre polynomials to the Bessel measure.
    """
    # The zeros of J_{1/2}(x) = sin(sqrt(x))/sqrt(x) are at x = (n*pi)^2 for n=1,2,...
    # These are positive, finite, and pairwise distinct.
    z = [(i * math.pi) ** 2 for i in range(1, 13)]
    return z
# EVOLVE-BLOCK-END
