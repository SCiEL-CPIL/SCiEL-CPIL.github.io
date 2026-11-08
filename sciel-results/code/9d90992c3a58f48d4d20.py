import numpy as np
from scipy.signal.windows import kaiser
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Generate a Kaiser window
    # Beta controls the sidelobe level. Higher beta = lower sidelobes.
    # Beta=10 gives approx -50dB sidelobes.
    beta = 10.0
    w = kaiser(N, beta)
    
    # Optional: Simple refinement if time permits
    # For now, the Kaiser window is a significant improvement over Uniform.
    
    return [float(x) for x in w]
