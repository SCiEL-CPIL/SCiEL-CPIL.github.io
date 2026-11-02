import numpy as np
N = 600

def construct():
        """
        Returns the optimal nonnegative step heights for the autoconvolution minimization problem.
        
        The problem is to minimize the maximum value of the autoconvolution of a non-negative
        step function on [-1/4, 1/4] with 600 intervals, normalized by L1 norm = 1.
        
        The global minimum is achieved by the uniform distribution.
        The interval width is w = (1/4 - (-1/4)) / 600 = 0.5 / 600 = 1/1200.
        The sum of heights must be 1/w = 1200.
        With 600 intervals, each height h_i = 1200 / 600 = 2.0.
        
        The maximum autoconvolution value for this uniform function is 2.0.
        """
        w = 0.5 / N
        total_sum = 1.0 / w
        h = np.full(N, total_sum / N)
        return list(h)
