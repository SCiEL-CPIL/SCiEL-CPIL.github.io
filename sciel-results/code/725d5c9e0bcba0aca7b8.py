import numpy as np

def construct():
    N = 400
    # Use a power law to minimize the peak of the autoconvolution
    # h_i = i^p
    # p=3 gives a good balance
    i = np.arange(N)
    h = i**3
    return h
