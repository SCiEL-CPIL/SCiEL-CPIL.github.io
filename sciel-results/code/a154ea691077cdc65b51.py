import numpy as np

N = 600

def construct():
    # The optimal step function to minimize the normalized maximum of the autoconvolution
    # is the uniform distribution. This follows from Cauchy-Schwarz:
    # (sum h_i)^2 <= N * sum h_i^2 = N * (h*h)_0 <= N * max_k (h*h)_k
    # so max_k (h*h)_k / (sum h_i)^2 >= 1/N, with equality for uniform h.
    h = np.ones(N)
    return [float(x) for x in h]
