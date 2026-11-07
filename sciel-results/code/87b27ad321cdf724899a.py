"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache to store the best parameters and basis matrix
# This ensures we don't recompute if the evaluator calls equation multiple times
# with the same data (though typically it's called once for train, once for test).
# However, the 'params' argument changes during BFGS, so we can't cache based on params.
# We CAN cache the basis matrix and the optimal params found via internal LSQ.

_cache = {
    'basis': None,
    'x': None,
    'params': None,
    'timestamp': 0
}

def _compute_basis(x):
    """
    Compute a diverse, well-scaled basis for symbolic regression.
    x is assumed to be in [-3, 3].
    
    Basis components:
    0: 1 (bias)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: x^5
    6: sin(x)
    7: cos(x)
    8: tanh(x)
    9: sech(x) = 1/cosh(x) (smooth, bounded)
    
    Scaling:
    - Polynomials up to x^5 can reach 243. 
    - To avoid ill-conditioning in the Gram matrix, we don't strictly need to scale 
      the basis if we use np.linalg.lstsq which handles SVD robustly. 
      However, normalizing columns to unit variance can help. 
      Let's just use raw values but ensure no overflow. x^5=243 is fine for float64.
    """
    x = np.asarray(x, dtype=float)
    # Ensure 1D
    if x.ndim != 1:
        x = x.flatten()
        
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    b5 = x**5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.tanh(x)
    # sech(x) is 1/cosh(x). cosh(3) ~ 10.06, so sech is in [0.1, 1].
    b9 = 1.0 / np.cosh(x)
    
    # Stack columns
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    return B

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Robust symbolic regression via internal Least Squares.
    
    Strategy:
    1. Check if we have cached the basis and optimal params for this specific x.
       Since x is fixed for the training phase and then fixed for the test phase,
       and the evaluator calls equation(x_train, params_iter) many times, 
       we can compute the optimal params ONCE for the training set.
       
       However, we don't know if it's train or test from x alone (unless we track calls).
       Actually, the evaluator fits params on TRAIN. So during training, x is x_train.
       During testing, x is x_test.
       
       We can cache the result for x_train. 
       When x changes (to x_test), we recompute the basis but reuse the params?
       No, the params are optimized for x_train. They should generalize to x_test.
       
       So:
       1. If x is new (different shape/values), compute Basis B.
       2. If we haven't solved for params for this x yet (training phase), 
          solve min ||B @ p - y||^2. But we don't have y!
          
    Wait, the signature is `equation(x, params) -> np.ndarray`.
    The evaluator passes `params` which it is currently optimizing.
    It expects `equation` to return predictions based on `x` and `params`.
    
    If I ignore `params` and use my own internal params, I am effectively 
    defining a model that doesn't depend on the input `params`.
    This is allowed as long as it returns the correct shape and finite values.
    The evaluator will minimize the loss over `params`, but if my function 
    ignores `params`, the gradient will be 0, and BFGS will stop immediately 
    or fail to converge, but the value returned will be based on MY internal fit.
    
    BUT, I need `y` to fit the internal params. I don't have `y` in the function signature.
    
    RE-READ TASK: "The evaluator fits params... minimizing the training MSE."
    "equation(x, params) -> np.ndarray"
    
    I cannot access `y`. Therefore, I CANNOT fit the parameters internally without `y`.
    The strategy of "internal LSQ" is invalid because I don't have the target values.
    
    CORRECTION:
    I must use the `params` provided by the evaluator.
    The problem with the current program is likely that the basis is ill-conditioned 
    or BFGS struggles.
    
    Alternative Strategy: 
    The current basis has high-order polynomials. 
    Let's try a basis that is **linearly independent** and **well-conditioned**.
    Chebyshev polynomials are orthogonal on [-1, 1]. Our x is in [-3, 3].
    We can map x to [-1, 1] and use Chebyshev polynomials.
    This ensures the Gram matrix is well-conditioned, making BFGS (or any optimizer) 
    converge much faster and more reliably to the global minimum.
    
    Additionally, we can include a few non-polynomial terms if they are also scaled.
    But mixing orthogonal polynomials with sines/cosines might break orthogonality.
    
    Let's stick to a pure polynomial basis using Chebyshev nodes/values for stability.
    Map x from [-3, 3] to t in [-1, 1]: t = x / 3.
    Basis: T_0(t), T_1(t), ..., T_9(t).
    
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    ...
    
    This basis is orthogonal with respect to the weight 1/sqrt(1-t^2).
    Even without the weight, they are well-conditioned compared to powers of x.
    
    Let's implement Chebyshev basis.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.flatten()
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials of the first kind
    # T_0 = 1
    # T_1 = t
    # T_{n+1} = 2*t*T_n - T_{n-1}
    
    T = np.zeros((len(x), 10))
    T[:, 0] = 1.0
    if len(x) > 0:
        T[:, 1] = t
        for n in range(2, 10):
            T[:, n] = 2.0 * t * T[:, n-1] - T[:, n-2]
    
    # The model is linear in params:
    # y = sum(params[i] * T[:, i])
    
    # This is still linear in params. BFGS will work well.
    # The conditioning is much better than x^0...x^9.
    
    # To further help BFGS, we can normalize the columns of T to have unit norm.
    # This makes the Hessian closer to identity (if the data is random).
    norms = np.linalg.norm(T, axis=0)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    T_scaled = T / norms
    
    # Now the model is:
    # y = sum( (params[i] / norms[i]) * T[:, i] )
    # Let alpha[i] = params[i] / norms[i]
    # y = sum(alpha[i] * T[:, i])
    
    # But the evaluator optimizes `params`. 
    # If I use T_scaled, the effective weight for the i-th basis function is params[i] / norms[i].
    # This is fine. The optimizer will find the best params[i] such that 
    # params[i]/norms[i] is the optimal coefficient for T[:, i].
    
    result = T_scaled @ params
    
    return result

# EVOLVE-BLOCK-END
