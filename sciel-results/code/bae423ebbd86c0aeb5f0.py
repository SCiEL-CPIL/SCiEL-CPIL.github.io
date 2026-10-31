import numpy as np

def construct():
    N = 100
    # The optimal solution is h_i = 0.5 for all i.
    # This satisfies:
    # 1. h_i in [0, 1]
    # 2. sum(h) = 100 * 0.5 = 50 = N/2
    # 3. Overlap O_k = sum h_i (1-h_{i+k}) = sum 0.5 * 0.5 = 25.
    # This is the theoretical minimum since average overlap is 25.
    h = np.full(N, 0.5)
    return h
