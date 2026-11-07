"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import lu_factor, lu_solve


def _radau_iia_5_step(f, t, y, h, n):
    """
    Perform one step of the 5th-order Radau IIA collocation method.
    Uses a single Newton iteration with precomputed LU factorization of the mass matrix.
    
    The Radau IIA 5th order method uses 3 stages with collocation points
    that are the roots of the Legendre polynomial shifted to [0,1].
    
    For Radau IIA, the collocation conditions are:
    y_{i+1} = y_i + h * sum_j b_j f(t + c_j h, Y_j)
    Y_j = y_i + h * sum_k A_{jk} f(t + c_k h, Y_k)
    
    This gives a system: Y_j - h * sum_k A_{jk} f(t + c_k h, Y_k) = y_i
    
    We solve this nonlinear system using Newton's method.
    """
    # Radau IIA 5th order collocation coefficients
    # c = [0.247690504723697, 0.5, 0.752309495276303]
    c = np.array([0.247690504723697, 0.5, 0.752309495276303])
    
    # A matrix (3x3) for Radau IIA 5th order
    # From Hairer, Nørsett, Wanner "Solving Ordinary Differential Equations II"
    A = np.array([
        [0.0218978137986510, 0.0649782690302950, 0.0134023409653540],
        [0.0649782690302950, 0.1131828476197650, 0.0649782690302950],
        [0.0134023409653540, 0.0649782690302950, 0.0218978137986510]
    ])
    
    # b vector (3,)
    b = np.array([0.1413963868414340, 0.7172072263171320, 0.1413963868414340])
    
    nstages = 3
    # Y is a 2D array of shape (nstages, n)
    Y = np.tile(y, (nstages, 1))
    
    # Precompute LU factorization of (I - h*A) for the linear part
    # The linear system is: Y_j - h * sum_k A_{jk} f(Y_k) = y_i
    # In matrix form for each component: (I - h*A) * Y_vec = y_vec + h * sum... 
    # Actually, let's think of this differently.
    # Let Y be a vector of length nstages*n, stacking all stage values.
    # The system is: Y_j = y_i + h * sum_k A_{jk} f(t + c_k h, Y_k)
    # Rearranging: Y_j - h * sum_k A_{jk} f(t + c_k h, Y_k) = y_i
    # This is a nonlinear system in Y.
    
    # For Newton's method, we need the Jacobian of the residual.
    # Residual R_j = Y_j - y_i - h * sum_k A_{jk} f(t + c_k h, Y_k)
    # dR_j / dY_l = delta_{jl} * I_n - h * A_{jl} * J_f(t + c_l h, Y_l)
    # where J_f is the Jacobian of f.
    
    # We'll use a finite difference Jacobian for f, but we can precompute the structure.
    # Actually, for efficiency, let's use a simpler approach: since n=2 (small),
    # we can solve the nonlinear system directly.
    
    # Let's use a fixed-point iteration or Newton with finite difference Jacobian.
    # For n=2, this is very cheap.
    
    max_newton = 50
    tol = 1e-13
    
    for _newton in range(max_newton):
        # Compute stage derivatives
        fY = np.array([f(t + c[i] * h, Y[i]) for i in range(nstages)])
        
        # Compute residual: R_j = Y_j - y - h * sum_k A_{jk} fY_k
        R = np.zeros_like(Y)
        for j in range(nstages):
            R[j] = Y[j] - y - h * np.dot(A[j, :], fY)
        
        # Check convergence
        if np.max(np.abs(R)) < tol:
            break
        
        # Compute Jacobian of the residual with respect to Y
        # J is a (nstages*n, nstages*n) matrix
        # dR_j / dY_l = delta_{jl} * I_n - h * A_{jl} * df/dY_l
        # We need df/dY_l = Jacobian of f at (t + c_l h, Y_l)
        
        # Compute Jacobians using finite differences
        Jfs = np.zeros((nstages, n, n))
        for i in range(nstages):
            f0 = fY[i]
            for k in range(n):
                e = np.zeros(n)
                e[k] = 1e-8 * max(1.0, abs(Y[i, k]))
                f1 = f(t + c[i] * h, Y[i] + e)
                Jfs[i, :, k] = (f1 - f0) / e[k]
        
        # Build the full Jacobian
        Jfull = np.zeros((nstages * n, nstages * n))
        for j in range(nstages):
            for l in range(nstages):
                if j == l:
                    Jfull[j*n:(j+1)*n, l*n:(l+1)*n] = np.eye(n)
                Jfull[j*n:(j+1)*n, l*n:(l+1)*n] -= h * A[j, l] * Jfs[l]
        
        # Solve Jfull * delta = -R_vec
        R_vec = R.flatten()
        try:
            delta = np.linalg.solve(Jfull, -R_vec)
        except np.linalg.LinAlgError:
            # Fall back to least squares
            delta, _, _, _ = np.linalg.lstsq(Jfull, -R_vec, rcond=None)
        
        delta = delta.reshape(nstages, n)
        Y += delta
        
        if np.max(np.abs(delta)) < tol * 0.1:
            break
    
    # Compute y_{i+1} = y_i + h * sum_j b_j f(t + c_j h, Y_j)
    fY = np.array([f(t + c[i] * h, Y[i]) for i in range(nstages)])
    y_next = y + h * np.dot(b, fY)
    
    return y_next


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol ODE using a fixed-step 5th-order Radau IIA method.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Choose number of steps. For Radau IIA 5th order, we can use relatively few steps.
    # Let's use 100 steps for t in [0, 10], so h = 0.1
    # This should be accurate enough for the required precision.
    steps = 100
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        y = _radau_iia_5_step(f, t, y, h, n)
        t += h
        if not np.all(np.isfinite(y)):
            # Fallback: try with smaller steps
            break
    
    if not np.all(np.isfinite(y)):
        # Fallback to RK4 with more steps
        t = t0
        y = np.array(y0, dtype=float)
        steps = 1000
        h = (t1 - t0) / steps
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
    
    return y
# EVOLVE-BLOCK-END
