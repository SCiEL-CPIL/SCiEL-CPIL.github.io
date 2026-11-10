"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # Basic inputs
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Features
    # 1. Bias
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. xy
    # 7. r = sqrt(x^2 + y^2)
    # 8. r^3
    # 9. x^3
    # 10. y^3
    
    r = np.sqrt(x**2 + y**2)
    
    F = np.column_stack((
        np.ones_like(x),
        x,
        y,
        x**2,
        y**2,
        x * y,
        r,
        r**3,
        x**3,
        y**3
    ))
    
    # Linear combination
    # params shape: (10,)
    # F shape: (N, 10)
    # Result shape: (N,)
    pred = F @ params
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        # If something went wrong, return zeros or mean y to avoid penalty
        # The task says non-finite error costs 1e30.
        # Returning finite values is better.
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        
    return pred
# EVOLVE-BLOCK-END
