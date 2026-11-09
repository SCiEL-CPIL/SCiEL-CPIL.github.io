# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Kakeya needle 2D positions for n=64.
    
    Uses the insight that to minimize union area, we want all triangle
    cross-sections to overlap as much as possible at each height.
    
    At height y, triangle i has left edge L_i = x_i + a_i*y where a_i=(i+1)/64.
    The spread of a_i*y across i is (63/64)*y. By choosing x_i = -c*a_i,
    we get L_i = a_i*(y-c), with spread (63/64)*|y-c|.
    
    Minimizing the integral of |y-c| over y in [0,1] gives c=0.5.
    So x_i = -0.5*(i+1)/64 = -(i+1)/128.
    
    This creates maximum overlap of all triangles around their mid-heights.
    """
    n = 64
    i = np.arange(n, dtype=np.float64)
    a = (i + 1) / n  # a_i = (i+1)/64
    
    # Optimal: x_i = -0.5 * a_i to center the overlap
    positions = -0.5 * a
    
    return positions.astype(np.float64)


# EVOLVE-BLOCK-END
