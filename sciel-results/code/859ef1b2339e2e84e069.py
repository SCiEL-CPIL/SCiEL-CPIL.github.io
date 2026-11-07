"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a self-contained neural network optimization.
    
    The provided `params` (length 10) are used to initialize a small neural network.
    We assume `x` contains the training inputs and we use `params` as a seed for 
    initialization. Since we don't have the targets `y` directly in the function call, 
    this strategy is fundamentally limited unless we assume the evaluator provides 
    more context or we exploit the fact that `params` might encode the solution 
    indirectly. However, a more robust interpretation of "different strategy" given 
    the constraints (no y access) is to use a **richer, fixed, non-linear basis** 
    that is linear in parameters, allowing BFGS to find the global optimum for a 
    much broader class of functions than simple linear or low-degree polynomials.
    
    The previous "inspiration" notes mention that a Chebyshev+Fourier basis allowed 
    BFGS to find a global optimum for a richer class. We will implement a hybrid 
    basis: 
    1. Chebyshev polynomials (up to degree 8) to capture smooth global trends.
    2. A few Gaussian RBFs to capture local features.
    
    Wait, the directive asks for a strategy MATERIALLY DIFFERENT from the current 
    program (linear). The inspirations show that fixed basis expansions work well 
    (score 641166 vs 1.0). The current program is linear. 
    A "neural network" approach that requires `y` is impossible because `y` is not 
    passed to `equation`. The evaluator calls `equation(x, params)` to get predictions. 
    The evaluator minimizes training MSE. It does NOT pass `y` to `equation`.
    
    Therefore, ANY strategy that tries to "learn" inside `equation` without `y` is 
    impossible. The only way to improve is to provide a better **basis function** 
    that is linear in the parameters, so that the external BFGS optimizer can 
    effectively fit it.
    
    The "different strategy" here is to move from a simple linear basis to a 
    **high-capacity, non-linear, linear-in-params basis** specifically designed to 
    be well-conditioned for BFGS and capable of approximating a wide variety of 
    symbolic regression targets (polynomials, trigonometric, exponential, etc.).
    
    We will use a combination of:
    - Scaled Chebyshev polynomials T_k(x/3) for k=0..7 (8 terms)
    - Two Gaussian RBFs centered at -1 and 1 with width 1.5 (2 terms)
    Total 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Normalize x to [-1, 1] for Chebyshev stability, since x is in [-3, 3]
    x_norm = x / 3.0
    
    # Build basis matrix
    # Terms 0-7: Chebyshev polynomials T_0 to T_7
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    T = np.zeros((8, len(x_norm)))
    T[0, :] = 1.0
    if len(x_norm) > 0:
        T[1, :] = x_norm
    for k in range(2, 8):
        T[k, :] = 2 * x_norm * T[k-1, :] - T[k-2, :]
    
    # Terms 8-9: Gaussian RBFs
    # Centered at -1 and 1 in original scale, width sigma=1.5
    sigma = 1.5
    rbf1 = np.exp(-0.5 * ((x - (-1.0)) / sigma) ** 2)
    rbf2 = np.exp(-0.5 * ((x - 1.0) / sigma) ** 2)
    
    # Let's re-implement cleanly
    basis = np.vstack([
        T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7],
        np.exp(-0.5 * ((x - 1.0) / 1.5) ** 2),
        np.exp(-0.5 * ((x + 1.0) / 1.5) ** 2)
    ])
    
    # Linear combination: y = params @ basis
    # params is shape (10,), basis is shape (10, N)
    # result is shape (N,)
    return basis.T @ params

# EVOLVE-BLOCK-END
