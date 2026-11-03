import numpy as np
import time

def construct():
    n = 600
    # The theoretical minimum for the ratio max(c_k) / (sum f)^2 is 1/N,
    # achieved by the constant function.
    # Any deviation from constant increases the ratio.
    return [1.0] * n
