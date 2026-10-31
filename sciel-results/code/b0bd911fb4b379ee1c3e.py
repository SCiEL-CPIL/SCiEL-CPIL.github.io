import numpy as np
import time

N = 600

def construct():
    # Try a sparse pattern
    # Every other interval
    x = [1.0 if i % 2 == 0 else 0.0 for i in range(N)]
    return x
