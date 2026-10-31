"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomial values at 0 (physicist's convention)
# H_0(0) = 1
# H_4(0) = -12
# H_8(0) = 1680
# H_12(0) = -498960
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -498960.0

def _compute_integrals():
    """
    Compute integrals I0[m,n] = ∫ e^{-x^2} H_m(x) H_n(x) dx
    and I2[m,n] = ∫ x^2 e^{-x^2} H_m(x) H_n(x) dx
    for m, n in {0, 4, 8, 12}.
    
    Using the fact that physicist's Hermite polynomials are orthogonal with weight e^{-x^2}:
    ∫ e^{-x^2} H_m(x) H_n(x) dx = sqrt(pi) * 2^m * m! * delta_{mn}
    
    For I2, we use the identity:
    x^2 H_n(x) = (1/2) H_{n+2}(x) + n H_n(x) + 2n(n-1) H_{n-2}(x)
    
    So ∫ x^2 e^{-x^2} H_m(x) H_n(x) dx can be computed from the orthogonality.
    """
    from scipy.special import eval_hermite
    
    degrees = [0, 4, 8, 12]
    idx = {d: i for i, d in enumerate(degrees)}
    
    # Use high-accuracy Gauss-Hermite quadrature
    # 200 points should be more than sufficient for polynomials up to degree 24
    x, w = np.polynomial.hermite.hermgauss(200)
    
    I0 = np.zeros((4, 4))
    I2 = np.zeros((4, 4))
    
    for i, m in enumerate(degrees):
        Hm = eval_hermite(m, x)
        for j, n in enumerate(degrees):
            Hn = eval_hermite(n, x)
            I0[i, j] = np.sum(w * Hm * Hn)
            I2[i, j] = np.sum(w * x**2 * Hm * Hn)
    
    return I0, I2

# Precompute the integral matrices
I0_mat, I2_mat = _compute_integrals()

# Signs for Fourier transform: c_ft[k] = c[k] * (-1)^(deg/2)
# degs: 0, 4, 8, 12 -> signs: 1, -1, 1, -1
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])

def _bound_and_grad(coeffs):
    """
    Compute the upper bound C4 = sqrt(V_x * V_xi) and its gradient.
    
    coeffs: array of shape (3,) with [c0, c4, c8]
    The c12 coefficient is derived from g(0) = 0 constraint.
    
    V_x = (c^T I2 c) / (c^T I0 c)
    V_xi = (c_ft^T I2 c_ft) / (c_ft^T I0 c_ft)
    
    where c_ft = c * signs
    """
    c0, c4, c8 = coeffs
    # c12 from g(0) = 0: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = (c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / (-_H12_0)
    
    c = np.array([c0, c4, c8, c12])
    c_ft = c * _SIGNS
    
    # Compute V_x and V_xi
    num_x = c @ I2_mat @ c
    den_x = c @ I0_mat @ c
    num_xi = c_ft @ I2_mat @ c_ft
    den_xi = c_ft @ I0_mat @ c_ft
    
    if den_x < 1e-15 or den_xi < 1e-15:
        return 1e10, np.zeros(3)
    
    V_x = num_x / den_x
    V_xi = num_xi / den_xi
    
    bound = np.sqrt(V_x * V_xi)
    
    # Gradient computation
    # V_x = num_x / den_x, dV_x/dc = (2*I2*c*den_x - num_x*2*I0*c) / den_x^2
    #     = 2*(I2*c*den_x - num_x*I0*c) / den_x^2
    # V_xi similarly with c_ft
    # dbound/dc = (1/(2*bound)) * (V_xi * dV_x/dc + V_x * dV_xi/dc)
    
    # dV_x/dc for the full c vector (4 components)
    dV_x_dc = 2.0 * (I2_mat @ c * den_x - num_x * (I0_mat @ c)) / (den_x * den_x)
    dV_xi_dc = 2.0 * (I2_mat @ c_ft * den_xi - num_xi * (I0_mat @ c_ft)) / (den_xi * den_xi) * _SIGNS
    
    # dbound/dc (4 components)
    if bound < 1e-15:
        grad_full = np.zeros(4)
    else:
        grad_full = (1.0 / (2.0 * bound)) * (V_xi * dV_x_dc + V_x * dV_xi_dc)
    
    # Project to 3 free parameters
    # c12 = (c0*1 + c4*(-12) + c8*1680) / 498960
    # dc12/dc0 = 1/498960, dc12/dc4 = -12/498960, dc12/dc8 = 1680/498960
    dc12_dc0 = _H0_0 / (-_H12_0)
    dc12_dc4 = _H4_0 / (-_H12_0)
    dc12_dc8 = _H8_0 / (-_H12_0)
    
    grad = np.array([
        grad_full[0] + grad_full[3] * dc12_dc0,
        grad_full[1] + grad_full[3] * dc12_dc4,
        grad_full[2] + grad_full[3] * dc12_dc8
    ])
    
    return bound, grad

def _objective(coeffs):
    val, _ = _bound_and_grad(coeffs)
    return val

def _objective_grad(coeffs):
    val, grad = _bound_and_grad(coeffs)
    return val, grad

def construct():
    t0 = time.time()
    deadline = t0 + 190  # Safety margin: 210s limit, leave 20s buffer
    
    # Seed initial guess
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_val, _ = _bound_and_grad(x0)
    best_x = x0.copy()
    
    # Multi-start with L-BFGS-B using analytic gradients
    np.random.seed(42)
    
    # Define several diverse starting points
    starts = [x0.copy()]
    
    # Add perturbations of the seed
    for scale in [0.5, 1.0, 2.0, 5.0]:
        for _ in range(3):
            x_rand = x0 * scale + 0.1 * scale * np.random.randn(3)
            starts.append(x_rand)
    
    # Add some random starts in a wider range
    for _ in range(10):
        x_rand = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.01, 0.01)
        ])
        starts.append(x_rand)
    
    # Try each starting point
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(_objective_grad, x_start, method='L-BFGS-B',
                          jac=True, options={'gtol': 1e-14, 'maxiter': 500, 'maxfun': 1000})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res_final = minimize(_objective_grad, best_x, method='L-BFGS-B',
                                jac=True, options={'gtol': 1e-16, 'maxiter': 1000, 'maxfun': 2000})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)

# EVOLVE-BLOCK-END
