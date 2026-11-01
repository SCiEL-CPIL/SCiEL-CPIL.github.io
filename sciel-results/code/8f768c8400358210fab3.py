import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient.
    X: (N, 3) array of points on the unit sphere.
    Returns:
        energy: float
        grad: (N, 3) array of Euclidean gradients.
    """
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Distances
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # We compute sum over all i!=j and divide by 2
    # To avoid log(0), we clip distances
    d_safe = np.maximum(dists, 1e-12)
    
    # Energy
    # sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # Let S = sum_{i!=j} log(d_ij). Then sum_{i<j} log(d_ij) = 0.5 * S.
    # So E = -0.5 * S.
    log_dists = np.log(d_safe)
    total_log_sum = np.sum(log_dists)
    energy = -0.5 * total_log_sum
    
    # Gradient
    # E = -0.5 * sum_{i!=j} log(||x_i - x_j||)
    # dE/dx_i = -0.5 * sum_{j!=i} (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j||
    #         = -0.5 * sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # 
    # Let's compute the gradient term by term.
    # inv_dists = 1 / dists (with diagonal 0)
    inv_dists = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_dists[mask] = 1.0 / dists[mask]
    
    # grad_i = -0.5 * sum_{j!=i} (x_i - x_j) * inv_dists[i,j]
    # diff[i,j] = x_i - x_j
    # We need sum over j for each i.
    # weighted_diff = diff * inv_dists[:, np.newaxis]
    weighted_diff = diff * inv_dists[:, np.newaxis]
    grad = -0.5 * np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _energy(X):
    """
    Compute only the energy for scoring/checking.
    """
    E, _ = _energy_and_grad(X)
    return E

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _optimize(X_init, time_limit=None):
    """
    Run L-BFGS-B on the flattened vector, projecting back to sphere?
    Actually, standard L-BFGS-B works on Euclidean space.
    We can minimize on the sphere by using the gradient projected onto the tangent space?
    Or simply minimize the energy on R^3 and normalize?
    The energy is defined on the sphere. If we allow points to move in R^3, the gradient points towards the sphere.
    However, L-BFGS-B is for unconstrained (or box-constrained) optimization.
    A common trick for sphere constraints is to use the gradient on the sphere directly.
    
    Let's use scipy.optimize.minimize with method='L-BFGS-B' on the flattened array.
    But the constraint ||x_i||=1 is non-linear.
    
    Alternative: Use a Riemannian optimizer if available? No, standard SciPy.
    
    Let's try minimizing the energy in R^3. The gradient of the energy with respect to x_i in R^3
    includes the term that pulls it to the sphere?
    Wait, the energy E(X) is only defined for unit vectors.
    If we define E(X) for any X by first normalizing? No, that's non-smooth.
    
    Better approach:
    The gradient of E on the sphere is the projection of the Euclidean gradient onto the tangent space.
    grad_sphere = grad_euclidean - (grad_euclidean . x_i) x_i
    
    If we use L-BFGS-B on the flattened vector, it will try to minimize the function.
    If we supply the gradient, it uses it.
    But L-BFGS-B doesn't know about the sphere constraint. It will move points off the sphere.
    
    So, we should use a method that handles the constraint, or we manually project.
    However, standard `minimize` doesn't easily support custom projection steps in the loop.
    
    Let's try a different approach:
    Use `scipy.optimize.minimize` with `method='BFGS'` or `L-BFGS-B` on the vector of angles?
    No, spherical coordinates have singularities.
    
    Let's stick to the idea of optimizing in R^3 but re-normalizing at each step?
    `scipy.optimize.minimize` doesn't allow custom callbacks to modify the variables during the iteration easily without hacking.
    
    Alternative:
    Implement a simple Riemannian Gradient Descent / L-BFGS manually?
    Or use `scipy.optimize.minimize` with `constraints`?
    Non-linear constraints are slow.
    
    Let's look at Inspiration 1: "deterministic, gradient-based Riemannian optimization (L-BFGS-B)".
    How is that done?
    Usually, one minimizes the energy on the manifold.
    If we flatten X to a vector v of length 3N.
    We can define a function that takes v, normalizes each 3-vector to get X, computes E and grad_X.
    Then we need grad_v.
    grad_v = dE/dv = dE/dX * dX/dv.
    Since X = v / ||v||, the projection of grad_X onto the tangent space is what matters for the direction on the sphere.
    The component of grad_X along v is irrelevant for moving on the sphere.
    So, we can just use the tangent gradient as the gradient for the optimizer?
    If we use an optimizer that assumes unconstrained variables, and we provide the tangent gradient,
    the optimizer will take a step in that direction.
    But the step will be in R^3.
    Then we re-normalize.
    This is effectively "Gradient Descent with Normalization".
    L-BFGS-B might not be the best fit if we are re-normalizing externally, because L-BFGS-B builds a model of the Hessian in R^3, but the function is actually defined on the sphere.
    
    However, for N=100, a simple Riemannian L-BFGS implementation or just using the standard L-BFGS-B with the "normalization trick" often works well enough if we treat the normalization as part of the function evaluation?
    No, if we normalize inside the function, the gradient is wrong (it's the gradient of the composed function, which includes the projection).
    
    Let's try this:
    1. Define `fun(v)`:
       - Reshape v to (N, 3)
       - Normalize rows to get X
       - Compute E and grad_X
       - Project grad_X to tangent space: grad_t = grad_X - (grad_X * X).sum(axis=1, keepdims=True) * X
       - Return E, grad_t
    2. Use `scipy.optimize.minimize` with `method='L-BFGS-B'`.
       The optimizer will take a step `delta = -alpha * grad_t` (roughly).
       The new `v_new = v + delta`.
       Then `fun(v_new)` will normalize it.
       This is a common heuristic. It works because the step is small.
    
    Let's implement this.
    
    """
    v_init = X_init.flatten()
    
    def fun_and_grad(v):
        X = v.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        X = X / norms
        
        E, grad_X = _energy_and_grad(X)
        
        # Project gradient to tangent space
        # grad_t_i = grad_X_i - (grad_X_i . X_i) X_i
        proj = np.sum(grad_X * X, axis=1, keepdims=True)
        grad_t = grad_X - proj * X
        
        return E, grad_t.flatten()
    
    # L-BFGS-B is good for large N.
    # maxiter: limit to prevent hanging.
    # We will call this multiple times with different starts.
    
    res = minimize(
        fun_and_grad, 
        v_init, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
    )
    
    X_opt = res.x.reshape(N, 3)
    X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
    
    return X_opt, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_X = None
    best_energy = float('inf')
    
    # Generate multiple starting configurations
    # 1. Fibonacci sphere
    # 2. Random starts
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    # Add random starts
    rng = np.random.RandomState(42)
    for _ in range(9):
        starts.append(_random_sphere(N, seed=rng.randint(10000)))
    
    for i, X_start in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        try:
            X_opt, E_opt = _optimize(X_start)
            
            # Verify energy
            E_check = _energy(X_opt)
            
            if E_check < best_energy:
                best_energy = E_check
                best_X = X_opt.copy()
                
        except Exception:
            continue
            
    # If we still have time, do a final polish on the best solution found
    if best_X is not None and time.time() < deadline - 10.0:
        try:
            X_polished, E_polished = _optimize(best_X)
            E_check = _energy(X_polished)
            if E_check < best_energy:
                best_energy = E_check
                best_X = X_polished.copy()
        except Exception:
            pass
            
    if best_X is None:
        # Fallback
        best_X = _fibonacci_sphere(N)
        
    # Final normalization
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
