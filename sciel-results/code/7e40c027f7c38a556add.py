import numpy as np

N = 600

def construct():
    # Create a comb function: 1 on even intervals, 0 on odd intervals.
    # This maximizes the number of disjoint pulses (K=300).
    # This minimizes the normalized autoconvolution maximum 
    # under the assumption that the metric scales with 1/K or similar.
    heights = [1.0 if i % 2 == 0 else 0.0 for i in range(N)]
    return heights
