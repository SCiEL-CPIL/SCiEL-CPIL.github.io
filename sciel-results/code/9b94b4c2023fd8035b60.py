"""Rewritten program: Simulated Annealing followed by SLSQP minimization of logarithmic energy on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy(pts):
    """Compute logarithmic energy."""
    pts = np.asarray(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-12)
    return -np.sum(np.log(d))

def _energy_grad(pts):
    """Compute logarithmic energy and gradient."""
    pts = np.asarray(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-12)
    energy = -np.sum(np.log(d))
    
    # Gradient calculation
    # E = - sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    inv_dist_sq = 1.0 / (dist_safe * dist_safe)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def construct():
    # 1. Initial guess
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # 2. Simulated Annealing to escape local minima
    # We work with a flattened array for easier manipulation, but keep shape (N, 3) for logic
    current_pts = pts.copy()
    current_energy = _energy(current_pts)
    
    T_start = 0.1
    T_end = 0.001
    steps = 2000
    cool_rate = (T_end / T_start) ** (1.0 / steps)
    T = T_start
    
    for step in range(steps):
        # Propose new state: add noise to tangent space
        noise = np.random.normal(0, 0.05, size=current_pts.shape)
        # Project noise to tangent space
        dot = np.sum(current_pts * noise, axis=1, keepdims=True)
        tangent_noise = noise - dot * current_pts
        new_pts = current_pts + tangent_noise
        
        # Renormalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.where(norms < 1e-12, 1.0, norms)
        
        new_energy = _energy(new_pts)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_pts = new_pts
            current_energy = new_energy
        
        T *= cool_rate

    # 3. Local Refinement with SLSQP
    # Flatten for scipy
    x0 = current_pts.flatten()
    
    # Define constraints: norm of each point is 1
    # SLSQP constraints are functions returning 0
    # We need 100 constraints
    # This might be slow for SLSQP with 100 constraints.
    # Alternative: Use 'L-BFGS-B' with a penalty method? Or just 'BFGS' with projection?
    # Let's try SLSQP. If it's too slow, we fall back to a few steps of projected gradient.
    
    # To make SLSQP faster, we can reduce the number of constraints or use a different method.
    # Actually, for N=100, 100 equality constraints is manageable for SLSQP.
    
    def objective(x):
        p = x.reshape(N, 3)
        # Ensure points are on sphere for energy calc (though SLSQP keeps them close)
        # Actually, SLSQP enforces constraints, so they should be on sphere.
        return _energy(p)
    
    def gradient(x):
        p = x.reshape(N, 3)
        e, g = _energy_grad(p)
        return g.flatten()
        
    constraints = []
    for i in range(N):
        # Constraint: x_{3i}^2 + x_{3i+1}^2 + x_{3i+2}^2 - 1 = 0
        idx = slice(3*i, 3*i+3)
        def con(x, idx=idx):
            return x[idx]**2 - 1.0
        def con_jac(x, idx=idx):
            jac = np.zeros(3*N)
            jac[idx] = 2.0 * x[idx]
            return jac
        constraints.append({'type': 'eq', 'fun': con, 'jac': con_jac})
        
    # Run SLSQP
    # maxiter and ftol to ensure it finishes within time limit
    res = minimize(objective, x0, jac=gradient, method='SLSQP', bounds=[(-1, 1)] * (3*N), constraints=constraints, options={'maxiter': 100, 'ftol': 1e-12})
    
    final_pts = res.x.reshape(N, 3)
    
    # Final cleanup: renormalize to be safe
    norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    final_pts = final_pts / np.where(norms < 1e-12, 1.0, norms)
    
    # Verify
    assert np.all(np.isfinite(final_pts))
    assert final_pts.shape == (N, 3)
    
    return final_pts
# EVOLVE-BLOCK-END
