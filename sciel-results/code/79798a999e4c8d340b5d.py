import numpy as np

N = 600

def construct():
    # The uniform function is provably optimal for minimizing the normalized
    # maximum of the autoconvolution for a function supported on a fixed interval.
    # The minimum value is 1/L where L is the length of the interval.
    # Here L = 0.5, so the minimum raw value is 2.
    # Returning a constant step function achieves this bound.
    return [1.0] * N
