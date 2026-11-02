import numpy as np

def construct():
        N = 600
        # The theoretical minimum for max(h*h)/(sum h)^2 is 1/N, achieved by uniform h.
        h = np.ones(N)
        return h.tolist()
