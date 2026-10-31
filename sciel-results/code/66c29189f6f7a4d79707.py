import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 170  # Safety margin before 180s limit

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1)/(2n)
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k is the coefficient of x^{2k} in P(x)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# Constraint P(0) = 0 => a0*1 + a4*3 + a8*105 + a12*945 = 0
# a12 = -(a0 + 3*a4 + 105*a8) / 945

# Matrix A such that p = A * a, where a = [a0, a4, a8]
# p has 7 components (k=0..6)

H0_c = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

sign_vec = np.array([1.0, -1, 1, -1, 1, -1, 1])

def _build_quadratic_forms():
    """
    Build the quadratic form matrices for Vx and Vxi.
    Vx = (a^T B1 a) / (a^T B0 a)
    Vxi = (a^T C1 a) / (a^T C0 a)
    """
    M0 = np.zeros((7, 7))
    M1 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            M0[i, j] = I[i + j]
            M1[i, j] = I[i + j + 1]
    
    D = np.diag(sign_vec)
    
    B0 = A_mat.T @ M0 @ A_mat
    B1 = A_mat.T @ M1 @ A_mat
    
    C0 = A_mat.T @ (D @ M0 @ D) @ A_mat
    C1 = A_mat.T @ (D @ M1 @ D) @ A_mat
    
    return B0, B1, C0, C1

# Precompute the matrices
B0, B1, C0, C1 = _build_quadratic_forms()

def _objective_and_grad(a):
    """
    Computes the uncertainty product J = sqrt(Vx * Vxi) and its gradient.
    """
    a = np.asarray(a)
    
    b0_val = a @ B0 @ a
    b1_val = a @ B1 @ a
    c0_val = a @ C0 @ a
    c1_val = a @ C1 @ a
    
    if b0_val <= 1e-15 or c0_val <= 1e-15:
        return 1e18, np.zeros(3)
    
    vx = b1_val / b0_val
    vxi = c1_val / c0_val
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
    
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient of J = 0.5 * (Vx * Vxi)^-0.5 * (Vxi * dVx + Vx * dVxi)
    # dVx = (2 B1 a * b0_val - b1_val * 2 B0 a) / b0_val^2
    # dVxi = (2 C1 a * c0_val - c1_val * 2 C0 a) / c0_val^2
    
    dB1a = B1 @ a
    dB0a = B0 @ a
    dC1a = C1 @ a
    dC0a = C0 @ a
    
    dvx = (2.0 * dB1a * b0_val - b1_val * 2.0 * dB0a) / (b0_val * b0_val)
    dvxi = (2.0 * dC1a * c0_val - c1_val * 2.0 * dC0a) / (c0_val * c0_val)
    
    grad = 0.5 / bound * (vxi * dvx + vx * dvxi)
    
    return bound, grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value for comparison
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _objective_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    # 1. Global Search using Differential Evolution
    # Bounds for a0, a4, a8
    # Based on seed and typical scales
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.05, 0.05)]
    
    if time.time() < _DEADLINE - 60:
        try:
            # DE is slow, so limit iterations
            res_de = differential_evolution(
                lambda x: _objective_and_grad(x)[0], 
                bounds, 
                maxiter=100, 
                tol=1e-12,
                popsize=15,
                seed=42,
                polish=False  # We will do our own polish
            )
            if np.isfinite(res_de.fun) and res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # 2. Local Polish from the best DE result
    if time.time() < _DEADLINE - 10:
        try:
            res_local = minimize(
                _objective_and_grad, 
                best_params, 
                jac=lambda x: _objective_and_grad(x)[1], 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if np.isfinite(res_local.fun) and res_local.fun < best_val:
                best_val = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
            
    # 3. Multi-start local optimization to ensure we didn't miss a better local min
    # Generate a few more starts around the found solution and the seed
    starts = [best_params.copy(), seed_params.copy()]
    
    # Add some random perturbations around the best found
    np.random.seed(123)
    for _ in range(10):
        start = best_params + np.random.normal(0, 0.01, 3)
        starts.append(start)
        
    # And around seed
    for _ in range(10):
        start = seed_params + np.random.normal(0, 0.01, 3)
        starts.append(start)
        
    for start in starts:
        if time.time() > _DEADLINE - 5:
            break
        try:
            res_local = minimize(
                _objective_and_grad, 
                start, 
                jac=lambda x: _objective_and_grad(x)[1], 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if np.isfinite(res_local.fun) and res_local.fun < best_val:
                best_val = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
            
    # 4. Final high-precision polish
    if time.time() < _DEADLINE - 2 and best_val < seed_val:
        try:
            res_polish = minimize(
                _objective_and_grad, 
                best_params, 
                jac=lambda x: _objective_and_grad(x)[1], 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-17}
            )
            if np.isfinite(res_polish.fun) and res_polish.fun < best_val:
                best_val = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
