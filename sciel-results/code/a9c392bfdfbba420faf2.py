"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient on R^3 (then projected to tangent space by caller or here).
    For point i: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Because d/dx_i [ -log|x_i - x_j| ] = -(x_i - x_j)/|x_i - x_j|^2
    Wait: E = -sum log d_ij. d/dx_i log d_ij = (x_i - x_j)/d_ij^2.
    So grad_i E = -sum_{j!=i} (x_i - x_j)/d_ij^2.
    Let's re-verify.
    E = sum_{i<j} -log(d_ij).
    dE/dx_i = sum_{j!=i} d/dx_i (-log(d_ij)) = - sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij
             = - sum_{j!=i} (x_i - x_j)/d_ij^2.
    Correct.
    """
    n = len(points)
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)  # avoid div by zero
    
    # dists_sq_inv = 1 / dists_sq
    dists_sq_inv = 1.0 / dists_sq
    
    # grad_raw[i] = - sum_{j!=i} diff[i,j] * dists_sq_inv[i,j]
    # diff is (n,n,3), dists_sq_inv is (n,n)
    # We want to sum over axis=1 (the j dimension)
    # Multiply diff by dists_sq_inv[..., np.newaxis]
    grad_raw = -np.sum(diff * dists_sq_inv[:, :, np.newaxis], axis=1)
    
    # Energy: sum of -log(d_ij) for i<j
    # d_ij = sqrt(dists_sq[i,j])
    # log(d_ij) = 0.5 * log(dists_sq[i,j])
    # E = - sum_{i<j} 0.5 * log(dists_sq[i,j])
    # Using math.fsum for precision as per contract? 
    # The contract says "pair terms summed with math.fsum".
    # For the gradient, we don't need fsum, just the value.
    # But to match the evaluator exactly, we should compute the value with fsum.
    # However, for optimization, numpy sum is fine for the objective function 
    # as long as it's consistent. But let's try to be close.
    # Actually, the evaluator computes the score. The optimizer just needs a smooth landscape.
    # The discrepancy between np.sum and math.fsum is tiny (1e-15) and won't affect the optimizer's path significantly.
    
    iu = np.triu_indices(n, k=1)
    log_dists_sq = np.log(dists_sq[iu])
    energy = -0.5 * np.sum(log_dists_sq)
    
    return energy, grad_raw


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from(points, deadline):
    """
    Run L-BFGS-B optimization on the sphere.
    We parameterize points in R^3 and use a penalty or projection?
    Standard L-BFGS-B doesn't handle manifolds directly.
    However, for sphere constraints, we can use a penalty method or 
    simply normalize after each step (which breaks gradient consistency slightly 
    but often works well for this problem).
    
    Better approach: Use the fact that the gradient on the manifold is the 
    projection of the Euclidean gradient onto the tangent space:
    grad_manifold = grad_euclidean - (grad_euclidean . x) * x
    Since x is on the sphere, x . x = 1.
    
    If we minimize the Euclidean energy E(x) with the constraint ||x||=1,
    the Lagrangian is L = E(x) + lambda(||x||^2 - 1).
    grad L = grad E + 2 lambda x = 0.
    So grad E is parallel to x at the optimum? No, grad E is normal to the manifold.
    Wait, at the constrained minimum, grad E must be orthogonal to the tangent space,
    meaning grad E is parallel to x (the normal).
    So the projected gradient grad_E - (grad_E . x) x should be zero.
    
    So, we can define an objective function that takes flattened R^3 coordinates,
    normalizes them, computes energy and projected gradient, and returns both.
    But L-BFGS-B expects the gradient of the objective with respect to the input variables.
    If we define f(x_flat) = E(normalize(x_flat)), then the chain rule applies.
    df/dx_flat = dE/dx_norm * dx_norm/dx_flat.
    This is messy.
    
    Alternative: Use a penalty method.
    f(x) = E(x) + mu * sum(||x_i||^2 - 1)^2
    grad f(x) = grad E(x) + 4 mu * sum(x_i * (||x_i||^2 - 1))
    This is smooth and works with standard L-BFGS-B.
    We can increase mu over time.
    
    Let's try the penalty method.
    """
    x0 = points.flatten()
    n = len(points)
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        # Normalize to stay near sphere? No, let penalty handle it.
        # But if x gets far from sphere, log(d) might be weird? 
        # d is chordal distance. If points go to infinity, d -> infinity, log(1/d) -> -infinity.
        # We want to minimize log(1/d) = -log(d).
        # If points are far apart, energy is low (more negative).
        # But we want them on the sphere.
        # The penalty term ||x||^2 - 1 will pull them back.
        
        # Compute energy and grad on the current x (not normalized)
        # But the energy definition assumes points are on the sphere?
        # "log(1/distance)" where distance is chordal.
        # If points are not on the sphere, the "distance" is still Euclidean.
        # But the problem is "Place 150 points on the unit sphere".
        # So the domain is the sphere.
        # If we use penalty, we are effectively minimizing over R^3 with a constraint.
        # The energy function should be the one defined on R^3: -sum log(||x_i - x_j||).
        
        energy, grad_euc = _log_energy_and_grad(x)
        
        # Penalty term
        norms_sq = np.sum(x * x, axis=1)
        penalty = np.sum((norms_sq - 1.0) ** 2)
        # grad penalty = sum 2*(norms_sq - 1)*2*x_i = 4 * sum((norms_sq - 1) * x_i)
        grad_penalty = 4.0 * x * (norms_sq - 1.0)[:, np.newaxis]
        
        mu = 10.0  # Penalty weight
        total_energy = energy + mu * penalty
        total_grad = grad_euc + mu * grad_penalty
        
        return total_energy, total_grad.flatten()
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
        )
        
        if time.time() > deadline:
            return points
            
        x_opt = res.x.reshape(n, 3)
        x_opt = _normalize(x_opt)
        
        # Verify improvement
        e_opt, _ = _log_energy_and_grad(x_opt)
        e_init, _ = _log_energy_and_grad(points)
        
        if e_opt < e_init:
            return x_opt
        else:
            return points
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)

    # 1. Evaluate initial Fibonacci
    try:
        e, _ = _log_energy_and_grad(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    rng = np.random.default_rng(42)

    # 2. Optimize the Fibonacci start
    if time.time() < deadline - 30:
        refined = _optimize_from(best_points, deadline)
        try:
            e, _ = _log_energy_and_grad(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
        except Exception:
            pass

    # 3. Multi-start with random initializations
    # We have ~100s left.
    # Each L-BFGS run might take 1-5s depending on convergence.
    # Let's try 20-50 restarts.
    
    num_restarts = 30
    for k in range(num_restarts):
        if time.time() > deadline - 10:
            break

        pts = _random_sphere(N, rng)
        
        # Quick check: if random is already better (unlikely), keep it
        e_init, _ = _log_energy_and_grad(pts)
        
        refined = _optimize_from(pts, deadline)
        
        try:
            e_opt, _ = _log_energy_and_grad(refined)
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = refined
        except Exception:
            continue

    # 4. Final polish on the best point found
    if time.time() < deadline - 5:
        # Run L-BFGS again on the best point found, maybe with more iterations
        # Or just return it.
        # Let's try one more optimization pass to ensure convergence.
        polished = _optimize_from(best_points, deadline)
        try:
            e_pol, _ = _log_energy_and_grad(polished)
            if e_pol < best_energy:
                best_energy = e_pol
                best_points = polished
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
