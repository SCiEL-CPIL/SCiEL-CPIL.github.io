import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    """
    Construct the sequence of step heights using a differentiable optimization
    approach to minimize the maximum overlap (soft-max of cross-correlations).
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    best_h = None
    best_score = np.inf
    
    # Multiple restarts with random binary initializations
    for restart in range(10):
        np.random.seed(restart)
        # Create a random binary sequence with exactly N/2 ones
        x0 = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        x0[indices] = 1.0
        
        beta = 50.0  # Sharper soft-max
        
        def loss_and_grad(x):
            # Calculate C_k for all k
            C = np.zeros(N)
            for k in range(1, N):
                shifted = np.roll(x, -k)
                C[k] = np.sum(x * (1.0 - shifted))
            
            # Soft-max loss
            C_max = np.max(C[1:])
            # Handle case where C_max is 0 or very small to avoid log(0) or instability
            if C_max < 1e-6:
                exp_C = np.zeros(N-1)
                sum_exp = 1.0
                loss = C_max
            else:
                exp_C = np.exp(beta * (C[1:] - C_max))
                sum_exp = np.sum(exp_C)
                loss = C_max + np.log(sum_exp) / beta
            
            # Gradient
            grad = np.zeros(N)
            for k in range(1, N):
                if C_max < 1e-6:
                    weight = 1.0 / (N-1) # Uniform weight if all are ~0
                else:
                    weight = beta * exp_C[k-1] / sum_exp
                grad += weight * (1.0 - np.roll(x, -k) - np.roll(x, k))
            
            return loss, grad

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
        ]
        bounds = [(0.0, 1.0)] * N

        try:
            res = minimize(
                loss_and_grad, 
                x0, 
                method='SLSQP', 
                jac=True, 
                constraints=constraints, 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            h = res.x
            # Final correction for exact sum
            h_sum = np.sum(h)
            if h_sum != N / 2.0:
                h += (N / 2.0 - h_sum) / N
            h = np.clip(h, 0.0, 1.0)
            
            # Calculate actual max overlap for this result
            C = np.zeros(N)
            for k in range(1, N):
                shifted = np.roll(h, -k)
                C[k] = np.sum(h * (1.0 - shifted))
            current_score = np.max(C[1:])
            
            if current_score < best_score:
                best_score = current_score
                best_h = h
                
        except Exception as e:
            continue
            
    if best_h is None:
        # Fallback to cosine if all fail
        i = np.arange(N)
        best_h = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
        best_h = np.clip(best_h, 0.0, 1.0)
        best_h += (N / 2.0 - np.sum(best_h)) / N
        best_h = np.clip(best_h, 0.0, 1.0)
        
    return best_h
