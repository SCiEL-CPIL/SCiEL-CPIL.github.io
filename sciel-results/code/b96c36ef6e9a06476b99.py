import numpy as np
from scipy.special import roots_genlaguerre

def construct():
        m = 12
        # Try alpha = 0
        alpha = 0
        x, w = roots_genlaguerre(m, alpha)
        return list(np.sort(x))
