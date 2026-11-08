"""
Minimize the normalized maximum of the autoconvolution of a nonnegative step function.
The function is defined by 600 heights on equal intervals covering [-1/4, 1/4].
The normalized maximum is typically defined as max(f*f) / (integral(f))^2.
For nonnegative f, max(f*f) = (f*f)(0) = integral(f^2).
Thus the quantity to minimize is integral(f^2) / (integral(f))^2.
By the Cauchy-Schwarz inequality, this ratio is minimized when f is constant over its support.
To minimize the ratio further, we must maximize the length of the support.
The maximum support length is the full interval [-1/4, 1/4], which has length 0.5.
The minimum value of the ratio is 1/0.5 = 2.
The uniform distribution achieves this minimum.
"""
import time
import numpy as np

N = 600
T = 0.5  # Total length of the interval [-1/4, 1/4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution.
    
    The optimal strategy is to use a uniform distribution over the entire interval.
    This minimizes the L2 norm squared relative to the L1 norm squared.
    """
    # The height for a uniform distribution such that integral(f) = 1 is 1/T.
    # However, the normalization (integral(f))^2 cancels out the scale.
    # So any constant height works. We choose h = 1.0 for simplicity, 
    # or h = 1/T if we want integral(f)=1.
    # The score depends on the ratio, so scale doesn't matter.
    # We return h = 1.0 for all intervals.
    h = 1.0
    return [h] * N
# EVOLVE-BLOCK-END
