import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a Neural Network with a single hidden layer of 10 ReLU units.
    
    y(x) = sum_{j=0}^{9} w_j * ReLU(w_in_j * x + b_j)
    
    Parameters (length 10):
    - Since we only have 10 parameters, we cannot fully parameterize a 10-unit hidden layer 
      (which would need 10 weights + 10 biases + 10 output weights = 30 params).
    - Instead, we use a structured approach:
        - The 10 params are used as the output weights w_j.
        - The hidden layer weights w_in_j and biases b_j are fixed to create a rich basis.
        - We define fixed centers for the ReLU 'kinks' similar to RBFs but with ReLU activation.
        - ReLU(z) = max(0, z).
        - Let z_j = x - c_j. Then ReLU(z_j) = max(0, x - c_j).
        - This creates a piecewise linear basis function that is zero for x < c_j and linear for x > c_j.
        - By summing such functions with different centers, we can approximate any continuous piecewise linear function.
        - To capture non-linearities better, we can also include a symmetric variant or scale.
        
    Actually, with only 10 params, a simple linear combination of 10 fixed basis functions is the best we can do.
    The previous RBF approach failed. Let's try a different basis: 
    Scaled and shifted ReLUs.
    
    Basis functions: phi_j(x) = ReLU(x - c_j)
    Centers c_j are evenly spaced in [0.1, 6].
    
    This is a 'hinge' basis. It is known to be universal for continuous functions if we allow many units.
    With 10 units, it's limited, but let's see if it fits better than Gaussians.
    
    Alternatively, we can use a mixture of ReLUs and their negatives to allow for 'humps'.
    But params are just 10 scalars.
    
    Let's stick to the linear combination of fixed basis functions, but change the basis from Gaussian to ReLU hinges.
    """
    x = np.asarray(x, dtype=float)
    
    n_basis = 10
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, n_basis)
    
    # Compute ReLU basis: max(0, x - c_j)
    # x shape: (N,), centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # ReLU activation
    basis = np.maximum(0, diff)
    
    # Linear combination with parameters
    # params shape: (10,)
    # y = basis @ params
    y = np.dot(basis, params)
    
    return y
