import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945 to ensure P(0)=0
# P(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

# Precompute the 3x3 moment matrices
# N_x[m,n] = sum_{i,j} A[i,m] A[j,n] I[i+j]
# M_x[m,n] = sum_{i,j} A[i,m] A[j,n] I[i+j+1]
# N_xi[m,n] = sum_{i,j} A[i,m] A[j,n] I[i+j] * q_factor? 
# The code used q = p * (-1)^k.
# Let's stick to the code's logic for correctness, but vectorize it.
# q_k = p_k (-1)^k.
# If p = A a, then q = D A a where D = diag(1, -1, 1, -1, 1, -1, 1).
# Then N_xi = (DA)^T W_0 (DA)
# M_xi = (DA)^T W_1 (DA)

D = np.diag(np.array([1, -1, 1, -1, 1, -1, 1]))
A_xi = D @ A_mat

# W_0[i,j] = I[i+j]
# W_1[i,j] = I[i+j+1]
W_0 = np.zeros((7, 7))
W_1 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        W_0[i, j] = I[i+j]
        W_1[i, j] = I[i+j+1]

N_x = A_mat.T @ W_0 @ A_mat
M_x = A_mat.T @ W_1 @ A_mat
N_xi = A_xi.T @ W_0 @ A_xi
M_xi = A_xi.T @ W_1 @ A_xi

def _compute_bound(a):
    """
    Compute the uncertainty bound for given coefficients a = [a0, a4, a8].
    """
    # Quadratic forms
    Nx = np.dot(a, N_x @ a)
    Mx = np.dot(a, M_x @ a)
    Nxi = np.dot(a, N_xi @ a)
    Mxi = np.dot(a, M_xi @ a)
    
    if Nx <= 1e-15 or Nxi <= 1e-15:
        return 1e18
        
    vx = Mx / Nx
    vxi = Mxi / Nxi
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    """
    Nx = np.dot(a, N_x @ a)
    Mx = np.dot(a, M_x @ a)
    Nxi = np.dot(a, N_xi @ a)
    Mxi = np.dot(a, M_xi @ a)
    
    if Nx <= 1e-15 or Nxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx / Nx
    vxi = Mxi / Nxi
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient of f(a) = sqrt( (a^T M_x a / a^T N_x a) * (a^T M_xi a / a^T N_xi a) )
    # Let R_x = a^T M_x a / a^T N_x a
    # Let R_xi = a^T M_xi a / a^T N_xi a
    # f = sqrt(R_x R_xi)
    # grad f = 1/(2 sqrt(R_x R_xi)) * ( R_xi grad R_x + R_x grad R_xi )
    
    # grad (a^T M a / a^T N a) = (2 a^T M / (a^T N a) - 2 (a^T M a) (a^T N) / (a^T N a)^2)
    # = 2/(a^T N a) * (a^T M - R (a^T N))
    
    gMx = 2.0 * (M_x @ a)
    gNx = 2.0 * (N_x @ a)
    grad_Rx = (gMx / Nx) - (Mx / (Nx * Nx)) * gNx
    
    gMxi = 2.0 * (M_xi @ a)
    gNxi = 2.0 * (N_xi @ a)
    grad_Rxi = (gMxi / Nxi) - (Mxi / (Nxi * Nxi)) * gNxi
    
    grad = (vxi * grad_Rx + vx * grad_Rxi) / (2.0 * bound)
    
    return bound, grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 5  # seconds
    
    # Use Differential Evolution for global search
    # Bounds: a0 in [-5, 5], a4 in [-1, 1], a8 in [-0.01, 0.01]
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    # We use a wrapper to ensure we don't exceed deadline
    # DE does not have a built-in time limit, so we run it for a fixed number of iterations
    # or catch an exception. 
    # Since _compute_bound is very fast now, we can run many iterations.
    
    try:
        # popsize=15, maxiter=1000 should be fast.
        # tol=1e-12
        res = differential_evolution(
            lambda a: _compute_bound(a), 
            bounds, 
            seed=42, 
            maxiter=2000, 
            tol=1e-12,
            popsize=15,
            polish=True,  # Uses L-BFGS-B on the best result
            init='latinhypercube'
        )
        
        if np.isfinite(res.fun) and res.fun < best_val:
            candidate_a = res.x
            val_check = _compute_bound(candidate_a)
            if val_check < best_val:
                best_val = val_check
                best_params = candidate_a.copy()
    except Exception:
        pass
    
    # Final polish with L-BFGS-B
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
            
    # Try a few more local optimizations from random points if time permits
    if time.time() < _DEADLINE - deadline_margin:
        rng = np.random.default_rng(123)
        for _ in range(50):
            if time.time() > _DEADLINE - deadline_margin:
                break
            start = np.array([
                rng.uniform(-5, 5),
                rng.uniform(-1, 1),
                rng.uniform(-0.01, 0.01)
            ])
            try:
                res = minimize(_compute_bound_and_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                if np.isfinite(res.fun) and res.fun < best_val:
                    candidate_a = res.x
                    val_check = _compute_bound(candidate_a)
                    if val_check < best_val:
                        best_val = val_check
                        best_params = candidate_a.copy()
            except Exception:
                pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
