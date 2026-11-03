import numpy as np
from scipy.signal import windows

N = 600

def construct():
    # Use a Blackman window to minimize sidelobes
    # scipy.signal.windows.blackman returns a window of length N
    w = windows.blackman(N)
    
    # Ensure non-negative (Blackman is non-negative for N > 2)
    w = np.maximum(w, 0)
    
    # The problem asks for a sequence of nonnegative step heights.
    # The "normalized maximum" score is likely max_sidelobe / main_lobe.
    # Returning the window directly should yield a much lower score than 1.0.
    
    return list(w)
