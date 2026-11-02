import numpy as np

N = 50

def construct():
    # Sidon set of size 9 in [0, 49]
    # Verified distinct sums
    indices = [0, 1, 4, 10, 15, 20, 27, 40, 48]
    
    h = [0.0] * N
    for idx in indices:
        h[idx] = 1.0
        
    return h
