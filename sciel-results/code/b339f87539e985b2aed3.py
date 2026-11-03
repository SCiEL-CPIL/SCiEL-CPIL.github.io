"""Improved program using robust gradient-based optimization for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    We minimize the product of the spatial variance and the frequency variance
    (which corresponds to the upper bound on the uncertainty constant C4).
    """
    # Constants derived from Hermite polynomials (Probabilists' He_n)
    # g(x) = sum a_n He_n(x) exp(-x^2/2)
    # Constraint: g(0) = 0 => a0*1 + a4*3 + a8*105 + a12*10395 = 0
    
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # Factorials n!
    f0 = 1.0
    f4 = 24.0
    f8 = 40320.0
    f12 = 479001600.0
    
    # Moments <x^2 He_n, He_n> = (2n+1) n!
    x2_0 = 1.0 * f0
    x2_4 = 9.0 * f4
    x2_8 = 17.0 * f8
    x2_12 = 25.0 * f12
    
    # Moments <He_n', He_n'> = n n!
    k_0 = 0.0
    k_4 = 4.0 * f4
    k_8 = 8.0 * f8
    k_12 = 12.0 * f12

    # Coefficients for a12 substitution
    c0 = H0_0 / H12_0
    c4 = H4_0 / H12_0
    c8 = H8_0 / H12_0

    def get_a12(a0, a4, a8):
        return -(c0 * a0 + c4 * a4 + c8 * a8)

    def objective_and_grad(coeffs):
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        # Norm squared N
        N = a0**2 * f0 + a4**2 * f4 + a8**2 * f8 + a12**2 * f12
        
        if N < 1e-12:
            return 1e15, np.zeros(3)
            
        # Spatial moment X
        X = a0**2 * x2_0 + a4**2 * x2_4 + a8**2 * x2_8 + a12**2 * x2_12
        
        # Frequency moment K
        K = a0**2 * k_0 + a4**2 * k_4 + a8**2 * k_8 + a12**2 * k_12
        
        val = (X * K) / (N * N)
        
        # Derivatives of a12
        da12_da0 = -c0
        da12_da4 = -c4
        da12_da8 = -c8
        
        # Derivatives of N, X, K
        dN_da0 = 2*a0*f0 + 2*a12*f12*da12_da0
        dN_da4 = 2*a4*f4 + 2*a12*f12*da12_da4
        dN_da8 = 2*a8*f8 + 2*a12*f12*da12_da8
        
        dX_da0 = 2*a0*x2_0 + 2*a12*x2_12*da12_da0
        dX_da4 = 2*a4*x2_4 + 2*a12*x2_12*da12_da4
        dX_da8 = 2*a8*x2_8 + 2*a12*x2_12*da12_da8
        
        dK_da0 = 2*a0*k_0 + 2*a12*k_12*da12_da0
        dK_da4 = 2*a4*k_4 + 2*a12*k_12*da12_da4
        dK_da8 = 2*a8*k_8 + 2*a12*k_12*da12_da8
        
        # Gradient of J = X K / N^2
        # dJ = (K dX + X dK) / N^2 - 2 (X K / N^2) dN / N
        # dJ = (K dX + X dK - 2 val dN) / N^2
        
        dJ_da0 = (K * dX_da0 + X * dK_da0 - 2 * val * dN_da0) / (N * N)
        dJ_da4 = (K * dX_da4 + X * dK_da4 - 2 * val * dN_da4) / (N * N)
        dJ_da8 = (K * dX_da8 + X * dK_da8 - 2 * val * dN_da8) / (N * N)
        
        grad = np.array([dJ_da0, dJ_da4, dJ_da8])
        
        return val, grad

    def objective_wrapper(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    def grad_wrapper(coeffs):
        _, grad = objective_and_grad(coeffs)
        return grad

    # Multi-start optimization with diverse initial guesses
    best_val = float('inf')
    best_x = None
    
    # Generate diverse starts
    starts = []
    # Fixed good guesses
    starts.append(np.array([-1.0, 0.1, 0.01]))
    starts.append(np.array([0.0, -1.0, 0.1]))
    starts.append(np.array([0.0, 0.0, -1.0]))
    starts.append(np.array([1.0, -0.5, 0.25]))
    starts.append(np.array([-0.5, 1.0, -0.2]))
    
    # Random starts in a reasonable range
    np.random.seed(42)
    for _ in range(20):
        x = np.random.uniform(-2.0, 2.0, 3)
        starts.append(x)
        
    # Also try points on the unit sphere in coefficient space
    for _ in range(10):
        v = np.random.normal(size=3)
        v = v / np.linalg.norm(v)
        starts.append(v)

    for start in starts:
        try:
            res = minimize(objective_wrapper, start, jac=grad_wrapper, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final polish with high precision if needed
    if best_x is not None:
        try:
            res = minimize(objective_wrapper, best_x, jac=grad_wrapper, method='BFGS',
                           options={'maxiter': 1000, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback
        best_x = np.array([-113/100, 1/25, 1/3240])
        
    return best_x

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
