"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global state to cache results and track time
_state = {
    'best_params': None,
    'best_x': None,
    'best_y': None,
    'best_mse': np.inf,
    'start_time': None,
    'is_first_call': True
}

def _compute_basis(x):
    """
    Compute a rich but well-scaled basis of functions.
    We include polynomials up to degree 8, and trigonometric/hyperbolic terms.
    Scaling is applied to keep the condition number manageable for BFGS/LSQ.
    """
    x = np.asarray(x, dtype=float)
    
    # Polynomials (scaled)
    # x in [-3, 3]
    # x^8 max is 6561. Scale by 1000 -> ~6.5
    # x^7 max is 2187. Scale by 100 -> ~22
    # x^6 max is 729. Scale by 10 -> ~73
    # x^5 max is 243. Scale by 1 -> 243 (might be a bit high, let's scale by 10 too)
    # Actually, let's just use raw polynomials up to 6 and scale higher ones.
    # Or better, use orthogonal polynomials? No, simple scaling is safer for BFGS if not perfectly orthogonal.
    
    # Let's try a basis that covers common symbolic regression patterns:
    # 0: 1
    # 1: x
    # 2: x^2 / 10
    # 3: x^3 / 10
    # 4: x^4 / 100
    # 5: x^5 / 100
    # 6: x^6 / 1000
    # 7: sin(x)
    # 8: cos(x)
    # 9: tanh(x)
    
    # Wait, the previous program used this and scored 4365.
    # The inspiration suggests a self-contained local search.
    # Let's implement a strategy where we perform an internal optimization 
    # of the parameters using the training data, and then use those 
    # optimized parameters for all subsequent calls (train and test).
    # This bypasses the evaluator's BFGS entirely (or rather, we provide 
    # a model that is effectively fixed once the first train call is made).
    # 
    # Strategy:
    # 1. On the first call (which is always on training data), we have access to y (targets) 
    #    indirectly? NO, the interface is equation(x, params) -> predictions. 
    #    We do NOT have access to y in the equation function.
    #    
    #    CORRECTION: The evaluator calls equation(x_train, params) to get predictions, 
    #    computes MSE, and passes gradient to BFGS. We do NOT see y_train.
    #    Therefore, we CANNOT do internal least squares fitting because we don't have y.
    #    
    #    So the "self-contained local search" inspiration must refer to something else, 
    #    or it was a failed attempt.
    #    
    #    Let's re-read the task: "The evaluator fits params... minimizing the training MSE."
    #    We only see x and params. We return predictions.
    #    
    #    So we are stuck with providing a model that is linear in params (or easily optimizable) 
    #    so that BFGS can find the optimal weights.
    #    
    #    The current program uses a linear basis. This is good.
    #    Why is the score 4365? 
    #    Seed score 1.0 implies the target is very simple, likely linear or quadratic.
    #    If the target is linear, a linear basis should recover it perfectly (MSE ~ noise^2).
    #    Normalized MSE = MSE / Var(y).
    #    If target is linear, Var(y) is significant. Noise variance is 0.1^2 = 0.01.
    #    If we fit perfectly, Test MSE ~ 0.01.
    #    If Var(y) is large (e.g., 10), Norm MSE ~ 0.001. Score ~ 1/0.001 = 1000?
    #    Wait, score is seed/value. If seed=1.0, value=1.0.
    #    If we get value=0.001, score = 1/0.001 = 1000.
    #    Current score 4365 means value ~ 1/4365 ~ 0.00023.
    #    This is already very good!
    #    
    #    Why would we want to improve? Maybe the target is NOT linear, but our linear basis 
    #    is overfitting or underfitting?
    #    Or maybe BFGS is not converging well due to ill-conditioning?
    #    
    #    Let's try a DIFFERENT basis. 
    #    Instead of polynomials, use a Fourier series basis? 
    #    Or Radial Basis Functions (RBFs)?
    #    
    #    RBFs are universal approximators. A linear combination of RBFs is linear in params.
    #    Basis: exp(-gamma * (x - center)^2)
    #    This is linear in the weights. BFGS will work well if the basis functions are not too collinear.
    #    
    #    Let's try a Gaussian Process-like basis (RBFs) with centers spread across [-3, 3].
    #    This might capture non-linearities better than polynomials if the target is wiggly.
    #    
    #    Basis:
    #    0: 1
    #    1: x
    #    2: exp(-(x-(-2.5))^2)
    #    3: exp(-(x-(-1.5))^2)
    #    4: exp(-(x-(-0.5))^2)
    #    5: exp(-(x-0.5)^2)
    #    6: exp(-(x-1.5)^2)
    #    7: exp(-(x-2.5)^2)
    #    8: sin(x)
    #    9: cos(x)
    
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    gamma = 1.0  # Width of the Gaussians
    
    basis = np.empty((len(x), 10))
    basis[:, 0] = 1.0
    basis[:, 1] = x
    for i, c in enumerate(centers):
        basis[:, 2+i] = np.exp(-gamma * (x - c)**2)
    basis[:, 8] = np.sin(x)
    basis[:, 9] = np.cos(x)
    
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # If we haven't started, record time
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
        
    # Check time limit
    elapsed = time.time() - _state['start_time']
    if elapsed > 140:
        # Return a safe fallback (e.g., mean of x or zero)
        # But we must return something. 
        # If we are in the test phase, we don't have y. 
        # We should just return the prediction from the current params, 
        # but quickly.
        pass
        
    # Compute basis
    B = _compute_basis(x)
    
    # Prediction is B @ p
    # Ensure p has shape (10,)
    if p.shape[0] != 10:
        # Fallback if params length is wrong (shouldn't happen)
        p = np.ones(10)
        
    result = B @ p
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
