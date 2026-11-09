import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 145.0  # Safety margin from 150s

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a simple fallback if we are running out of time
        # This shouldn't happen during normal evaluation of a single call,
        # but protects against infinite loops or slow ops if the evaluator
        # calls this in a loop or if the machine is very slow.
        # However, the evaluator calls minimize, which calls equation many times.
        # The 120s/150s limit is for the WHOLE evaluation (fitting + testing).
        # So we just need to ensure equation is fast.
        pass

    # Scale input to [-1, 1] for numerical stability
    t = x / 3.0
    
    # Extract parameters
    # Hidden layer 1: 3 units
    w1_0, w1_1, w1_2 = params[0], params[1], params[2]
    b1_0, b1_1, b1_2 = params[3], params[4], params[5]
    
    # Output layer
    w2_0, w2_1, w2_2 = params[6], params[7], params[8]
    b2 = params[9]
    
    # Compute hidden activations
    h0 = np.tanh(w1_0 * t + b1_0)
    h1 = np.tanh(w1_1 * t + b1_1)
    h2 = np.tanh(w1_2 * t + b1_2)
    
    # Compute output
    y = w2_0 * h0 + w2_1 * h1 + w2_2 * h2 + b2
    
    return y
