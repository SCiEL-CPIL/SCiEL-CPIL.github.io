import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    """
    Construct the sequence of step heights using a differentiable optimization
    approach to minimize the maximum overlap (soft-max of cross-correlations).
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    # Initial guess: Cosine wave (from previous solution)
    i = np.arange(N)
    x0 = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    x0 = np.clip(x0, 0.0, 1.0)
    # Normalize to sum N/2
    x0 += (N / 2.0 - np.sum(x0)) / N
    x0 = np.clip(x0, 0.0, 1.0)
    
    beta = 20.0  # Sharpness of the soft-max
    
    def loss_and_grad(x):
        # Calculate C_k for all k
        C = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(x, -k)
            C[k] = np.sum(x * (1.0 - shifted))
        
        # Soft-max loss
        # To avoid overflow, subtract max
        C_max = np.max(C[1:])
        exp_C = np.exp(beta * (C[1:] - C_max))
        sum_exp = np.sum(exp_C)
        loss = C_max + np.log(sum_exp) / beta
        
        # Gradient
        # dL/dx_j = beta * sum_k (exp(beta(C_k - C_max)) / sum_exp) * dC_k/dx_j
        # dC_k/dx_j = 1 - x_{j-k} - x_{j+k}
        grad = np.zeros(N)
        for k in range(1, N):
            weight = beta * exp_C[k-1] / sum_exp  # exp_C is indexed 0..N-2 for k=1..N-1
            # dC_k/dx
            # x_j appears in term i=j (as x_i) and i=j-k (as x_{i+k})
            # d/dx_j [ x_j (1 - x_{j+k}) ] = 1 - x_{j+k}
            # d/dx_j [ x_{j-k} (1 - x_j) ] = -x_{j-k}
            # So dC_k/dx_j = 1 - x_{(j+k)%N} - x_{(j-k)%N}
            
            # Vectorized calculation for this k
            # We need to add weight * (1 - x_{j+k} - x_{j-k}) to grad[j]
            # This is equivalent to:
            # grad += weight * (1 - roll(x, -k) - roll(x, k))
            grad += weight * (1.0 - np.roll(x, -k) - np.roll(x, k))
            
        return loss, grad

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N

    res = minimize(
        loss_and_grad, 
        x0, 
        method='SLSQP', 
        jac=True, 
        constraints=constraints, 
        bounds=bounds, 
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    h = res.x
    # Final correction for exact sum
    h_sum = np.sum(h)
    if h_sum != N / 2.0:
        h += (N / 2.0 - h_sum) / N
    h = np.clip(h, 0.0, 1.0)
    
    return h
