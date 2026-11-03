import numpy as np
from scipy.optimize import minimize
import time

N = 600
L = 0.5  # Length of interval [-1/4, 1/4]
M = 30   # Number of harmonics

def construct():
    # Grid for evaluating f(x) >= 0
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    
    # Grid for evaluating g(t) = (f*f)(t)
    n_t = 4096
    t = np.linspace(-L, L, n_t, endpoint=False)
    
    # Precompute basis functions
    # f(x) = sum_{k=0}^M u_k * phi_k(x)
    # phi_0(x) = 1/sqrt(L)
    # phi_k(x) = sqrt(2/L) * cos(2*pi*k*x/L) for k >= 1
    # This basis is orthonormal on [-L/2, L/2]
    
    # Build matrix Phi where Phi[j, k] = phi_k(x_j)
    Phi = np.zeros((n_grid, M+1))
    Phi[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        Phi[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # For g(t), we need:
    # g(t) = (1/L) * [u_0^2 + sum_{k=1}^M u_k^2 * cos(2*pi*k*t/L)]
    # Let's precompute the cosine terms for g
    # G[k, j] = cos(2*pi*k*t_j/L) for k=1..M, j=0..n_t-1
    G = np.zeros((M, n_t))
    for k in range(1, M+1):
        G[k-1, :] = np.cos(2*np.pi*k*t/L)
    
    def objective(u):
        """
        Minimize max(g(t)) subject to:
        1. ||u||^2 = 1 (energy constraint)
        2. f(x) >= 0 for all x
        """
        # Compute g(t)
        g = np.zeros(n_t)
        g += u[0]**2 / L
        # g += sum_{k=1}^M (u_k^2 / L) * cos(2*pi*k*t/L)
        # u[1:] are u_1..u_M
        # G[k-1, :] corresponds to k
        for k in range(1, M+1):
            g += (u[k]**2 / L) * G[k-1, :]
        
        max_g = np.max(g)
        
        # Penalty for f(x) < 0
        f_vals = Phi @ u
        neg_f = np.maximum(0, -f_vals)
        penalty = np.sum(neg_f**2)
        
        return max_g + 100.0 * penalty
    
    def gradient(u):
        """
        Gradient of objective.
        """
        # Compute g(t)
        g = np.zeros(n_t)
        g += u[0]**2 / L
        for k in range(1, M+1):
            g += (u[k]**2 / L) * G[k-1, :]
        
        # Find index of max g
        idx_max = np.argmax(g)
        t_max = t[idx_max]
        
        # Gradient of max(g) w.r.t u
        # d/d u_k of g(t_max)
        # g(t_max) = u_0^2/L + sum_{k=1}^M u_k^2/L * cos(2*pi*k*t_max/L)
        grad_g = np.zeros(M+1)
        grad_g[0] = 2 * u[0] / L
        for k in range(1, M+1):
            grad_g[k] = 2 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
        
        # Gradient of penalty
        f_vals = Phi @ u
        grad_pen = np.zeros(M+1)
        for j in range(n_grid):
            if f_vals[j] < 0:
                # d/d u of (-f_j)^2 = -2 f_j * (d f_j / d u)
                # d f_j / d u = Phi[j, :]
                grad_pen -= 2 * f_vals[j] * Phi[j, :]
        
        return grad_g + 100.0 * grad_pen
    
    # Initial guess: uniform function
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    # Constraints
    # We need ||u||^2 = 1. We'll handle this by normalizing u after each step.
    # For scipy.optimize.minimize, we can use a penalty method or just project.
    # Let's use a simple projected gradient descent.
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Multiple restarts
    n_restarts = 10
    n_iter = 500
    lr = 0.01
    
    for restart in range(n_restarts):
        if restart == 0:
            u = u0.copy()
        else:
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
            # Ensure initial f is not too negative
            # If it is, we'll rely on the penalty to fix it
        
        for i in range(n_iter):
            f_vals = Phi @ u
            g = np.zeros(n_t)
            g += u[0]**2 / L
            for k in range(1, M+1):
                g += (u[k]**2 / L) * G[k-1, :]
            
            idx_max = np.argmax(g)
            t_max = t[idx_max]
            
            grad_g = np.zeros(M+1)
            grad_g[0] = 2 * u[0] / L
            for k in range(1, M+1):
                grad_g[k] = 2 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
            
            grad_pen = np.zeros(M+1)
            # Vectorized penalty gradient
            neg_mask = f_vals < 0
            if np.any(neg_mask):
                # grad_pen = -2 * sum_{j: f_j<0} f_j * Phi[j, :]
                # We can do this as:
                # weights = np.where(f_vals < 0, -2*f_vals, 0)
                # grad_pen = Phi.T @ weights
                weights = np.where(f_vals < 0, -2*f_vals, 0)
                grad_pen = Phi.T @ weights
            
            grad_total = grad_g + 100.0 * grad_pen
            
            u = u - lr * grad_total
            
            # Project to sphere ||u|| = 1
            norm_u = np.linalg.norm(u)
            if norm_u > 1e-12:
                u /= norm_u
            
            # Check if feasible
            f_check = Phi @ u
            if np.min(f_check) >= -1e-8:
                g_check = np.zeros(n_t)
                g_check += u[0]**2 / L
                for k in range(1, M+1):
                    g_check += (u[k]**2 / L) * G[k-1, :]
                score = np.max(g_check)
                if score < best_score:
                    best_score = score
                    best_u = u.copy()
    
    # Convert best_u to step heights
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Compute f at centers
    heights = np.zeros(N)
    heights += best_u[0] / np.sqrt(L)
    for k in range(1, M+1):
        heights += best_u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*centers/L)
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    return list(heights)
