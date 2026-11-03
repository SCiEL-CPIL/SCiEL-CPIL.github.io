import numpy as np
def construct():
    """
    Return the uniform distribution, which is the global minimum for this problem.
    The minimum value of the normalized max autoconvolution is 1/N.
    """
    N = 600
    f = np.ones(N) / N
    return [float(x) for x in f]
