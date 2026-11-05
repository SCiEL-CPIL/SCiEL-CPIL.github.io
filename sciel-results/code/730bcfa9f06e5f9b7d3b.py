import numpy as np
from scipy.optimize import minimize

N = 400
L = 2 * N - 1
target = 1.0 / L

def objective(h):
    # h is flattened array of size N
    c = np.convolve(h, h)
    # c has size L
    diff = c - target
    return np.sum(diff**2)

def gradient(h):
    # Gradient of sum((c_k - target)^2)
    # c_k = sum_i h_i h_{k-i}
    # This is quadratic in h.
    # Let's just use numerical gradient or automatic differentiation if possible?
    # SciPy minimize can handle numerical gradients.
    # For N=400, numerical gradient is 400 convolutions.
    # 400 * 400 * 2 operations ~ 320k ops. Very fast.
    pass

def construct():
    # Initial guess: random
    np.random.seed(42)
    h0 = np.random.randn(N)
    h0 = h0 / np.sum(h0)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    
    # Bounds: none, real numbers
    # But to prevent explosion, maybe add loose bounds?
    # The problem is scale invariant in a way, but we fixed sum=1.
    # If h gets large, c gets large, objective increases. So it should be bounded.
    
    try:
        res = minimize(objective, h0, method='L-BFGS-B', constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        h_opt = res.x
        # Ensure sum is exactly 1 to avoid floating point drift affecting score
        h_opt = h_opt / np.sum(h_opt)
        
        # Calculate score to verify
        c = np.convolve(h_opt, h_opt)
        max_c = np.max(c)
        value = 2 * N * max_c / (np.sum(h_opt)**2)
        # print(f"Optimized value: {value}")
        
        return h_opt
        
    except Exception as e:
        # Fallback to seed if optimization fails
        return np.ones(N)
