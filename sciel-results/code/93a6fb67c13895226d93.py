import numpy as np
from scipy.optimize import minimize

N = 600
L = 0.5  # Interval length [-1/4, 1/4]

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution (upper bound on C1).
    
    Approach:
    1. Represent f(x) using first M Fourier cosine coefficients (even, symmetric).
    2. f(x) = a0 + 2 * sum_{k=1}^{M} ak * cos(2*pi*k*x/L)
    3. The autoconvolution g(t) = (f*f)(t) for periodic extension has:
       g(t) = (1/L) * [a0^2 + sum_{k=1}^{M} ak^2 * cos(2*pi*k*t/L)]
    4. Energy E = L*a0^2 + 2*L*sum(ak^2)
    5. Minimize max(g(t)) / E subject to f(x) >= 0
    
    Using orthonormal basis:
    u0 = a0*sqrt(L), uk = ak*sqrt(2*L) for k>=1
    Then E = sum(uk^2), and we fix ||u||=1.
    
    f(x) = u0/sqrt(L) + sum_{k=1}^{M} uk*sqrt(2/L)*cos(2*pi*k*x/L)
    g(t) = (1/L) * [u0^2 + sum_{k=1}^{M} uk^2 * cos(2*pi*k*t/L)]
    
    We minimize max(g(t)) subject to ||u||=1 and f(x)>=0.
    """
    M = 20  # Number of harmonics
    n_x = 2048  # Grid size for checking f(x) >= 0
    n_t = 4096  # Grid size for computing max of g(t)
    
    # Precompute grids
    x = np.linspace(-L/2, L/2, n_x, endpoint=False)
    t = np.linspace(-L, L, n_t, endpoint=False)
    
    # Precompute cosine matrices
    # C_x[i, k] = cos(2*pi*k*x[i]/L) for k=1..M
    k_range = np.arange(1, M+1)
    C_x = np.cos(2*np.pi * np.outer(x, k_range) / L)  # shape (n_x, M)
    C_t = np.cos(2*np.pi * np.outer(t, k_range) / L)  # shape (n_t, M)
    
    def compute_f(u):
        """Compute f(x) values from coefficient vector u."""
        f_vals = np.full(n_x, u[0] / np.sqrt(L))
        f_vals += C_x @ (u[1:] * np.sqrt(2.0/L))
        return f_vals
    
    def compute_g(u):
        """Compute g(t) values from coefficient vector u."""
        g_vals = np.full(n_t, u[0]**2 / L)
        g_vals += C_t @ (u[1:]**2 / L)
        return g_vals
    
    def objective_and_grad(u):
        """Objective: max(g(t)) + penalty for f(x)<0. Returns (value, gradient)."""
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        
        # Penalty for negative f
        neg_mask = f_vals < 0
        penalty = 100.0 * np.sum(np.maximum(0, -f_vals)**2)
        
        # Find max of g
        idx_max = np.argmax(g_vals)
        t_max = t[idx_max]
        max_g = g_vals[idx_max]
        
        obj = max_g + penalty
        
        # Gradient
        grad = np.zeros(M+1)
        
        # Gradient of max_g w.r.t u
        # g(t_max) = u0^2/L + sum(uk^2/L * cos(2*pi*k*t_max/L))
        # d/d u0 = 2*u0/L
        # d/d uk = 2*uk/L * cos(2*pi*k*t_max/L)
        grad[0] = 2.0 * u[0] / L
        for k in range(1, M+1):
            grad[k] = 2.0 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
        
        # Gradient of penalty
        if np.any(neg_mask):
            # penalty = 100 * sum(max(0, -f_j)^2) for j where f_j < 0
            # d/d u = 100 * sum_{j: f_j<0} (-2*f_j) * df_j/du
            # f_j = u0/sqrt(L) + sum_k uk*sqrt(2/L)*C_x[j,k]
            # df_j/du0 = 1/sqrt(L)
            # df_j/du_k = sqrt(2/L)*C_x[j,k]
            neg_f = np.maximum(0, -f_vals)
            # Weighted sum: sum_j neg_f[j] * f_basis_j
            # This is: (neg_f @ C_x) for k>=1, and sum(neg_f) for k=0
            weights = neg_f * 2.0  # because d(-f_j)^2/du = -2*f_j*df_j/du, and we have -f_j > 0
            grad[0] += 100.0 * np.sum(weights) / np.sqrt(L)
            grad[1:] += 100.0 * (weights @ C_x) * np.sqrt(2.0/L)
        
        return obj, grad
    
    # Initial guess: uniform function
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Multiple restarts
    n_restarts = 15
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if restart == 0:
            u = u0.copy()
        else:
            # Random initialization with DC dominant
            u = np.random.randn(M+1)
            # Make DC coefficient positive and large
            u[0] = np.abs(u[0]) + 0.8
            # Scale other coefficients smaller
            u[1:] *= 0.3
            # Normalize
            norm = np.linalg.norm(u)
            if norm > 0:
                u /= norm
        
        # Optimize
        try:
            result = minimize(
                objective_and_grad,
                u,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-6}
            )
            
            u_opt = result.x
            
            # Check feasibility
            f_check = compute_f(u_opt)
            if np.all(f_check >= -1e-5):
                g_check = compute_g(u_opt)
                score = np.max(g_check)
                if score < best_score:
                    best_score = score
                    best_u = u_opt.copy()
        except Exception:
            pass
    
    # Convert best_u to step heights
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Compute f at centers
    k_range_centers = np.arange(1, M+1)
    C_centers = np.cos(2*np.pi * np.outer(centers, k_range_centers) / L)
    
    heights = np.full(N, best_u[0] / np.sqrt(L))
    heights += C_centers @ (best_u[1:] * np.sqrt(2.0/L))
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    return list(heights)
