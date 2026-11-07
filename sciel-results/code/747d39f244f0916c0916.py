"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a normalized linear basis.
    
    The model is linear in the 10 parameters: y = sum(params[i] * norm_basis[i](x)).
    The basis functions are chosen to cover a wide range of functional forms:
    polynomials, reciprocals, logarithms, exponentials, trigonometric, and Gaussian.
    
    Crucially, each basis function is normalized (zero mean, unit variance) 
    over the input domain [0.1, 6] to ensure stable convergence of BFGS 
    starting from all-ones. This prevents scale imbalances between terms 
    (e.g., x^3 vs 1/x) from hindering the optimizer.
    """
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 100

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 1.0)
        
    n = x.shape[0]
    if n == 0:
        return np.array([])

    # Safe x for log and reciprocal
    x_safe = np.clip(x, 1e-10, None)
    
    # Compute raw basis terms
    # 0: Constant
    b0 = np.ones_like(x)
    
    # 1: Linear
    b1 = x.copy()
    
    # 2: Quadratic
    b2 = x * x
    
    # 3: Cubic
    b3 = b2 * x
    
    # 4: Reciprocal
    b4 = 1.0 / x_safe
    
    # 5: Logarithmic
    b5 = np.log(x_safe)
    
    # 6: Exponential (clipped to avoid overflow, scaled to be comparable)
    # exp(x) for x in [0.1, 6] is [1.1, 403]. Let's use exp(x/3) to keep it tighter? 
    # Or just clip. exp(6) is fine.
    b6 = np.exp(np.clip(x, -10, 10))
    
    # 7: Sine (frequency 2)
    b7 = np.sin(2.0 * x)
    
    # 8: Cosine (frequency 2)
    b8 = np.cos(2.0 * x)
    
    # 9: Gaussian RBF at center 3.0, sigma 1.0
    diff = x - 3.0
    b9 = np.exp(-0.5 * diff * diff)
    
    # Stack into matrix (n, 10)
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Check for non-finite values in basis
    if not np.all(np.isfinite(B)):
        B = np.where(np.isfinite(B), B, 0.0)
        
    # Normalize each column (basis function) to have zero mean and unit variance
    # This is crucial for BFGS stability with mixed scales.
    # We compute mean and std over the current batch of points.
    # Note: This makes the model slightly adaptive to the input distribution,
    # but since inputs are uniform [0.1, 6], this is consistent.
    
    means = np.mean(B, axis=0)
    stds = np.std(B, axis=0)
    
    # Avoid division by zero
    stds = np.where(stds < 1e-10, 1.0, stds)
    
    B_norm = (B - means) / stds
    
    # Final check for non-finite values
    if not np.all(np.isfinite(B_norm)):
        B_norm = np.where(np.isfinite(B_norm), B_norm, 0.0)
        
    # Prediction: linear combination
    prediction = B_norm @ p
    
    # Check for non-finite values in prediction
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
