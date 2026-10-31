import time
import sys

N = 600

# EVOLVE-BLOCK-START
def construct():
    # The problem asks to minimize the normalized maximum of the autoconvolution.
    # Based on the analysis, the ratio ||f*f||_inf / ||f||_1^2 is minimized by a box function
    # spanning the maximum possible width.
    # The maximum width on [-1/4, 1/4] is 1/2.
    # The optimal shape is a constant height function on this interval.
    # Height is arbitrary due to scaling invariance of the ratio, so we use 1.0.
    
    # Construct the box function
    # Height 1.0 on all N intervals
    result = [1.0] * N
    
    return result
# EVOLVE-BLOCK-END
