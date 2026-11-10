"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time
from scipy.special import lpmv
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.special import lpmv
from scipy.optimize import minimize

N = 150
DEADLINE_FRACTION = 0.85

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _energy_and_grad(pts):
    """Compute log energy and tangent-space gradient on the sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} -log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    energy = np.sum(-np.log(d))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    grads = np.sum(diff * inv_d2[:, np.newaxis, :], axis=1)
    
    # Project to tangent space
    for i in range(n):
        g = grads[i]
        p = pts[i]
        grads[i] = g - np.dot(g, p) * p
    
    return energy, grads

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_harmonic_init(n, rng, max_l=6):
    """
    Initialize points using a perturbed spherical harmonic basis.
    We start from a near-uniform distribution and add low-frequency
    perturbations to break symmetry and find better local minima.
    """
    # Start with Fibonacci
    pts = _fibonacci_spiral(n)
    
    # Add low-frequency noise
    # This is a heuristic to break the rigid symmetry of the Fibonacci spiral
    # which might trap it in a local minimum.
    noise = rng.standard_normal((n, 3)) * 0.1
    pts = pts + noise
    pts = _normalize(pts)
    return pts

def _optimize_lbfgs(pts, deadline):
    """Optimize using L-BFGS-B with a time budget."""
    best_pts = _normalize(pts.copy())
    best_energy, _ = _energy_and_grad(best_pts)
    
    def objective_and_grad(x):
        p = _normalize(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    # Wrap for scipy
    def obj(x):
        e, _ = objective_and_grad(x)
        return e
    
    def grad(x):
        _, g = objective_and_grad(x)
        return g
    
    x0 = best_pts.flatten()
    
    # Use a tighter tolerance and more iterations if time permits
    # But we need to be careful with the time limit.
    # L-BFGS-B can be slow. We'll use a wrapper that checks time.
    # Since scipy doesn't easily support callbacks for time checking in minimize 
    # without stopping, we'll just run it and hope it finishes or use a 
    # simpler gradient descent if L-BFGS is too slow.
    
    # Actually, for 150 points (450 dims), L-BFGS-B should be okay if we limit iterations.
    # Let's try a few iterations.
    
    try:
        # We'll run L-BFGS-B
        # maxls=10 to keep line searches short
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 10}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
    except Exception:
        pass
    
    return best_pts, best_energy

def _refine_with_newton(pts, deadline):
    """
    Attempt a Newton-like refinement.
    Since the Hessian is expensive (450x450), we might skip full Newton.
    Instead, we do a few steps of gradient descent with adaptive step size.
    """
    best_pts = _normalize(pts.copy())
    best_energy, _ = _energy_and_grad(best_pts)
    
    lr = 1e-3
    for step in range(50):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(best_pts)
        
        # Check for stagnation
        if np.linalg.norm(grad) < 1e-8:
            break
        
        # Gradient descent step
        new_pts = best_pts - lr * grad
        new_pts = _normalize(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            best_pts = new_pts
            best_energy = new_energy
            lr *= 1.05 # Slightly increase learning rate if we made progress
        else:
            lr *= 0.5 # Decrease learning rate if we failed
            if lr < 1e-6:
                break
                
    return best_pts, best_energy

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Start with a high-quality symmetric init (Fibonacci).
    # 2. Run a robust local optimizer (L-BFGS-B) to get to a local minimum.
    # 3. Use "symmetry breaking" restarts: perturb the best solution slightly
    #    and re-optimize. This helps escape shallow local minima.
    # 4. Use a "spectral" check: ensure we aren't stuck in a configuration
    #    with high low-frequency energy components (though for log energy,
    #    uniformity is key).
    
    # Initial phase: Optimize Fibonacci
    best_pts = _fibonacci_spiral(N)
    best_pts, best_energy = _optimize_lbfgs(best_pts, deadline)
    
    # Secondary phase: Refine with gradient descent (cheap, fast)
    best_pts, best_energy = _refine_with_newton(best_pts, deadline)
    
    # Tertiary phase: Symmetry breaking restarts
    # We take the current best solution and perturb it in the tangent space
    # to jump to a nearby basin of attraction.
    
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Perturbation magnitude decreases as we iterate? 
        # Or keep it constant?
        # Let's try a moderate perturbation.
        # The "Spherical Harmonic" inspiration suggests looking at global modes.
        # A perturbation of ~0.1 radians should be enough to jump basins
        # if the landscape is rugged, but not so much as to lose the good structure.
        
        # Generate random tangent vectors
        noise = rng.standard_normal((N, 3))
        # Project to tangent space
        for k in range(N):
            noise[k] = noise[k] - np.dot(noise[k], best_pts[k]) * best_pts[k]
        
        # Scale noise
        scale = 0.1 * (1.0 - i / (n_restarts + 1.0)) # Annealing the perturbation
        perturbed_pts = best_pts + scale * noise
        perturbed_pts = _normalize(perturbed_pts)
        
        # Optimize the perturbed version
        candidate_pts, candidate_energy = _optimize_lbfgs(perturbed_pts, deadline)
        
        # Refine
        candidate_pts, candidate_energy = _refine_with_newton(candidate_pts, deadline)
        
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy
    
    # Final polish
    if time.time() < deadline:
        best_pts, best_energy = _refine_with_newton(best_pts, deadline)
        
    return best_pts
# EVOLVE-BLOCK-END
