import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time

# Hermite polynomials (Physicists')
def H0(x): return np.ones_like(x)
def H4(x): return 16*x**4 - 48*x**2 + 12
def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
def H12(x):
    return 4096*x**12 - 73728*x**10 + 552960*x**8 - 2165760*x**6 + 3944640*x**4 - 2850240*x**2 + 665280

hs = [H0, H4, H8, H12]
ns = [0, 4, 8, 12]
h_vals_0 = np.array([h(0) for h in hs])

# Precompute grids for numerical integration of L1 norms
# The function decays as e^{-x^2/2}, so a range of [-10, 10] is sufficient.
x_grid = np.linspace(-10, 10, 2000)
dx = x_grid[1] - x_grid[0]
gauss = np.exp(-x_grid**2 / 2.0)

def get_coefficients(c):
    c0, c4, c8 = c
    # c12 is derived from g(0) = 0
    # c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    c12 = -(c0 * h_vals_0[0] + c4 * h_vals_0[1] + c8 * h_vals_0[2]) / h_vals_0[3]
    return np.array([c0, c4, c8, c12])

def true_objective(c):
    coeffs = get_coefficients(c)
    
    # f(x) = sum c_i H_i(x) e^{-x^2/2}
    f = np.zeros_like(x_grid)
    for i in range(4):
        f += coeffs[i] * hs[i](x_grid) * gauss
        
    # |f|_1
    l1_f = np.sum(np.abs(f)) * dx
    
    # Fourier Transform
    # If f(x) = P(x) e^{-x^2/2}, then F[f](xi) is related to P and derivatives.
    # Alternatively, we can compute the FT numerically or analytically.
    # For Hermite functions psi_n(x) = H_n(x) e^{-x^2/2}, 
    # F[psi_n](xi) = sqrt(2*pi) * (-i)^n psi_n(xi) (depending on convention).
    # Let's assume the standard unitary convention or the one used in the task.
    # Usually, the product ||f||_1 ||f_hat||_1 is invariant under scaling if we adjust constants,
    # but the relative weights of c_i matter.
    # Let's use the property: F[H_n(x) e^{-x^2/2}] = sqrt(2*pi) (-i)^n H_n(x) e^{-x^2/2}
    # Wait, the variable in the FT is xi.
    # f_hat(xi) = sum c_i sqrt(2*pi) (-i)^i H_i(xi) e^{-xi^2/2}
    
    f_hat = np.zeros_like(x_grid)
    for i in range(4):
        # (-i)^i
        phase = (-1j)**i
        f_hat += coeffs[i] * np.sqrt(2*np.pi) * phase * hs[i](x_grid) * gauss
        
    l1_fhat = np.sum(np.abs(f_hat)) * dx
    
    return l1_f * l1_fhat

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin
    
    # Initial guess: The seed coefficients
    # The seed program (score 1.0) likely used the initial guess or a specific known good point.
    # Let's try the initial guess from the current program:
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Also try a "pure Gaussian" like solution
    # If we want f close to Gaussian, c0 dominates.
    # Constraint: c0 + 12 c4 + 1680 c8 + 665280 c12 = 0
    # If c4=c8=0, c12 = -c0/665280.
    # Let's try c0=1, c4=0, c8=0.
    x1 = np.array([1.0, 0.0, 0.0])
    
    best_val = float('inf')
    best_x = x0
    
    # Evaluate initial candidates
    for x_init in [x0, x1]:
        val = true_objective(x_init)
        if val < best_val:
            best_val = val
            best_x = x_init
            
    # Use differential evolution for global search
    # Bounds for c0, c4, c8
    # c0 ~ 1, c4 ~ 0.01, c8 ~ 0.001
    bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
    
    try:
        res = differential_evolution(
            true_objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8,
            popsize=15,
            polish=True
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Local polish with Nelder-Mead
    if time.time() < deadline:
        try:
            res_local = minimize(
                true_objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_x = res_local.x
        except Exception:
            pass
            
    return np.array(best_x)
