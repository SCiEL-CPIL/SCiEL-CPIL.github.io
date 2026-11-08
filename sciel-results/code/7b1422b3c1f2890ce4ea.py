import numpy as np
from scipy.optimize import minimize


def construct():
    # Hermite polynomial values at 0 for even degrees (physicists' convention)
    # H_0(0) = 1, H_4(0) = -3, H_8(0) = 105, H_12(0) = -10395
    h0, h4, h8, h12 = 1.0, -3.0, 105.0, -10395.0
    
    # n + 1/2 for degrees 0, 4, 8, 12
    moments = np.array([0.5, 4.5, 8.5, 12.5])
    
    # Coefficients for the linear constraint g(0) = 0
    # c12 = -(c0*h0 + c4*h4 + c8*h8) / h12
    # Let A = [h0/h12, h4/h12, h8/h12]
    # c12 = - dot(c, A) where c = [c0, c4, c8]
    A = np.array([h0/h12, h4/h12, h8/h12])
    
    # We want to minimize (s/q)^2 which is equivalent to minimizing s/q
    # s = sum(c_i^2 * w_i) + c12^2 * w12
    # q = sum(c_i^2) + c12^2
    # c12 = -A^T c
    # q = c^T (I + A A^T) c
    # s = c^T (W + w12 A A^T) c
    # where W = diag(w0, w4, w8)
    
    W = np.diag(moments[:3])
    w12 = moments[3]
    
    C_mat = np.eye(3) + np.outer(A, A)
    B_mat = W + w12 * np.outer(A, A)
    
    # Solve generalized eigenvalue problem B v = lambda C v
    # We want the smallest eigenvalue
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(C_mat) @ B_mat)
        # Find index of smallest real eigenvalue
        idx = np.argmin(eigvals.real)
        best_vec = eigvecs[:, idx].real
    except Exception:
        # Fallback to previous method if linear algebra fails
        def objective(coeffs):
            c0, c4, c8 = coeffs
            c12 = -(c0 * h0 + c4 * h4 + c8 * h8) / h12
            c = np.array([c0, c4, c8, c12])
            c2 = c**2
            q = np.sum(c2)
            s = np.dot(c2, moments)
            if q < 1e-14:
                return 1e10
            return (s / q)**2
        
        initial_points = [
            np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),
            np.array([1.0, 0.0, 0.0]),
        ]
        best_x = None
        best_f = np.inf
        for x0 in initial_points:
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 200000,
                                       'adaptive': True})
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        if best_x is not None:
            return np.array(best_x)
        else:
            return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

    # Normalize the vector? The ratio is scale invariant, but returning a unit vector is standard.
    norm = np.linalg.norm(best_vec)
    if norm > 1e-10:
        best_vec = best_vec / norm
        
    return best_vec
