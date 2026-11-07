"""
Symbolic regression for a nonlinear oscillator using a Radial Basis Function (RBF) basis.
This approach uses fixed centers in the (x, v) space to create local basis functions.
The evaluator's BFGS fitter will optimize the linear combination of these RBFs.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard: evaluator kills at 150s, keep safety margin
_DEADLINE = time.time() + 130

def _time_left() -> float:
    return _DEADLINE - time.time()

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target variable using a Radial Basis Function model.
    
    Inputs:
        x: np.ndarray, position (shape: N,)
        v: np.ndarray, velocity (shape: N,)
        params: np.ndarray, 10 fitted constants (shape: 10,)
        
    Returns:
        np.ndarray, predictions (shape: N,)
    """
    try:
        # Ensure inputs are 1D arrays
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        
        # If params is not length 10, we can't proceed correctly
        if len(params) != 10:
            return np.zeros_like(x)
        
        # Define fixed centers for the RBFs in the (x, v) space.
        # The training range is x in [-0.5728, 0.5275], v in [-0.1754, 0.2821].
        # We place 9 centers in a grid pattern covering this domain, plus 1 constant term.
        # This provides local flexibility to fit complex nonlinear shapes.
        
        # Grid of centers: 3x3 grid
        # X centers: -0.4, 0.0, 0.4 (approx)
        # V centers: -0.1, 0.0, 0.1 (approx)
        # Let's use slightly wider spread to ensure coverage.
        
        centers_x = np.array([-0.4, 0.0, 0.4])
        centers_v = np.array([-0.1, 0.0, 0.1])
        
        # Compute distances from each point (x, v) to each center
        # We need to compute 9 distances.
        # To avoid memory issues with large N, we process in chunks or use vectorization efficiently.
        
        # params[0:9] are weights for the 9 RBFs
        # params[9] is the constant bias
        
        pred = np.zeros_like(x)
        
        # RBF width (sigma). 
        # A smaller sigma makes the basis more local. 
        # Given the domain size is ~1.1 in x and ~0.45 in v, a sigma of 0.2-0.3 seems appropriate.
        # Let's use a fixed sigma of 0.25 for stability.
        sigma = 0.25
        
        # Iterate over the 9 centers
        # This loop is over 9 items, which is fast.
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                cx = centers_x[i]
                cv = centers_v[j]
                
                # Squared Euclidean distance
                dx = x - cx
                dv = v - cv
                dist_sq = dx**2 + dv**2
                
                # Gaussian RBF
                rbf = np.exp(-dist_sq / (2 * sigma**2))
                
                pred += params[idx] * rbf
        
        # Add constant bias
        pred += params[9]
        
        # Ensure finite output
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred

    except Exception:
        # Fallback to a simple linear model if anything goes wrong
        try:
            return params[0]*x + params[1]*v + params[2]
        except Exception:
            return np.zeros_like(x)

# EVOLVE-BLOCK-END
