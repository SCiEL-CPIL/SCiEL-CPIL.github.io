"""Symbolic regression with 2 inputs using a rich Fourier basis + BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin before 120s limit

    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Construct a rich Fourier basis with 10 terms.
    # The domain is [-1, 1], so we use frequencies that are reasonable for this range.
    # We include:
    # - Constant
    # - Single variable sines/cosines with frequency 1 and 2
    # - Interaction terms (x+y, x-y) with frequency 1
    
    # Term 0: Constant (1)
    term_0 = np.ones_like(x)
    
    # Term 1: cos(x)
    term_1 = np.cos(x)
    
    # Term 2: sin(x)
    term_2 = np.sin(x)
    
    # Term 3: cos(y)
    term_3 = np.cos(y)
    
    # Term 4: sin(y)
    term_4 = np.sin(y)
    
    # Term 5: cos(2*x) - second harmonic in x
    term_5 = np.cos(2 * x)
    
    # Term 6: sin(2*x) - second harmonic in x
    term_6 = np.sin(2 * x)
    
    # Term 7: cos(2*y) - second harmonic in y
    term_7 = np.cos(2 * y)
    
    # Term 8: sin(2*y) - second harmonic in y
    term_8 = np.sin(2 * y)
    
    # Term 9: cos(x + y) - interaction
    term_9 = np.cos(x + y)
    
    # Stack terms into a matrix of shape (n_points, 10)
    basis = np.column_stack([
        term_0, term_1, term_2, term_3, term_4,
        term_5, term_6, term_7, term_8, term_9
    ])
    
    # The model is linear in params: pred = basis @ params
    # BFGS will optimize params to minimize MSE.
    # Since it's linear, this is effectively a least squares problem,
    # but BFGS is robust enough to find it quickly from any start.
    
    predictions = basis @ params
    
    # Ensure output is finite and correct shape
    if not np.all(np.isfinite(predictions)):
         # Fallback to zero if something went wrong (should not happen)
         return np.zeros_like(x)
        
    return predictions


# EVOLVE-BLOCK-END
