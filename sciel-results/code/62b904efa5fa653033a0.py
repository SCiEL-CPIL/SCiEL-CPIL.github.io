"""
Logarithmic energy minimization of 150 points on the unit sphere.

Strategy:
  - Multi-start basin-hopping (Fibonacci, random, perturbed).
  - Riemannian optimization: exponential-map steps and tangent-space L-BFGS
    polish that respect the sphere geometry exactly.
  - Velocity-Verlet and Adam phases for basin finding and refinement.
  - Local perturbation restarts using Riemannian perturbations.
  - Final high-precision polish pass.
"""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------
def _energy_and_grad(pts):
    """
    Logarithmic energy and tangent-space gradient.
    pts: (N, 3) unit vectors.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i (Cartesian) = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts - pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (fsum for precision)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


# ---------------------------------------------------------------------------
# Riemannian (exponential map) utilities
# ---------------------------------------------------------------------------
def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    which is equivalent to rotating x by angle |v| about the axis v/|v|.
    This respects the sphere geometry exactly.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    # Where norm ~ 0, c ~ 1, s ~ 0, so result ~ x
    out = c * pts + s * u
    return _project_to_sphere(out)


def _log_map_tangent(pts, target):
    """
    Logarithmic map: tangent vector at pts that maps to target on the sphere.
    v = theta * (target - cos(theta) x) / sin(theta), theta = arccos(x . target).
    """
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    # Avoid division by zero
    safe_sin = np.where(sin_theta > 1e-12, sin_theta, 1.0)
    v = (theta / safe_sin) * (target - cos_theta * pts)
    # Zero out where theta ~ 0
    v = np.where((theta > 1e-12)[:, np.newaxis], v, 0.0)
    return v


# ---------------------------------------------------------------------------
# Riemannian L-BFGS polish (optimizes in tangent space)
# ---------------------------------------------------------------------------
def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    We represent the configuration as a flat tangent vector t (size 3N) and
    define f(t) = energy(exp(pts, t)), grad_f(t) = grad of energy at exp(pts,t)
    pulled back (which for the sphere exponential map with the metric is just
    the tangent gradient, since the exponential map is an isometry in the
    infinitesimal sense for our purposes here).

    We use a trust-region-like approach: start from t=0 (current config),
    optimize t, and if it improves, set pts = exp(pts, t_best) and repeat.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(t):
        # t is flat tangent vector (3N,). Interpret as (N,3) tangent.
        tv = t.reshape(N, 3)
        # Make sure tangent is actually tangent (subtract radial part)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts)
        # Gradient of f w.r.t. t: since exp is local isometry, g (tangent) is
        # the gradient in t-space directly.
        return e, g.flatten()

    t0 = np.zeros(3 * N)
    try:
        res = minimize(fun, t0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        if time.time() < deadline:
            tv = res.x.reshape(N, 3)
            dot = np.sum(tv * pts, axis=1, keepdims=True)
            tv = tv - dot * pts
            new_pts = _exp_map_tangent(pts, tv)
            new_energy = _energy_only(new_pts)
            if new_energy < best_energy and np.all(np.isfinite(new_pts)):
                best_energy = new_energy
                best_pts = new_pts.copy()
    except Exception:
        pass

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Cartesian L-BFGS polish (fallback / final)
# ---------------------------------------------------------------------------
def _lbfgs_polish(pts, deadline, maxiter=500, ftol=1e-16, gtol=1e-14):
    """L-BFGS-B polish using flat Cartesian coordinates with projection."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_res = _energy_only(p_res)
        if e_res < best_energy and np.all(np.isfinite(p_res)):
            best_energy = e_res
            best_pts = p_res.copy()
    except Exception:
        pass
    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    Provides a different structure than the Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces until we have enough points
    while len(verts) < n:
        new_verts = []
        new_faces = []
        edge_midpoints = {}
        
        for face in faces:
            a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
            
            # Get or create midpoints
            for i, j in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                key = (min(i, j), max(i, j))
                if key not in edge_midpoints:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints[key] = len(verts) + len(new_verts)
                    new_verts.append(mid)
            
            m01 = edge_midpoints[(min(face[0], face[1]), max(face[0], face[1]))]
            m12 = edge_midpoints[(min(face[1], face[2]), max(face[1], face[2]))]
            m20 = edge_midpoints[(min(face[2], face[0]), max(face[2], face[0]))]
            
            new_faces.append([face[0], m01, m20])
            new_faces.append([face[1], m12, m01])
            new_faces.append([face[2], m20, m12])
            new_faces.append([m01, m12, m20])
        
        verts = np.vstack([verts] + [v.reshape(1, -1) for v in new_verts])
        faces = new_faces
        
        # Stop if we have enough
        if len(verts) >= n:
            break
    
    # Select n points (with some randomization if needed)
    if len(verts) > n:
        indices = np.random.choice(len(verts), n, replace=False)
        verts = verts[indices]
    
    return _project_to_sphere(verts)


def _random_sphere(n, seed=None):
    """Generate n random points uniformly on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


# ---------------------------------------------------------------------------
# Velocity-Verlet dynamics on the sphere
# ---------------------------------------------------------------------------
def _velocity_verlet_sphere(X, dt=0.01, max_iter=200, tol=1e-10, deadline=None):
    """
    Velocity-Verlet dynamics on the sphere to explore basins.
    Returns (X_final, energy_final).
    """
    X_curr = X.copy()
    V = np.zeros_like(X_curr)  # velocity in R^3
    energy_curr, grad_curr = _energy_and_grad(X_curr)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space (already done in _energy_and_grad)
        # grad_curr is already tangent
        
        # Velocity Verlet: half step
        V = V - 0.5 * dt * grad_curr
        # Project velocity to tangent space
        V_dot_X = np.sum(V * X_curr, axis=1)
        V = V - V_dot_X[:, np.newaxis] * X_curr
        
        # Full position step using exponential map
        X_new = _exp_map_tangent(X_curr, dt * V)
        
        energy_new, grad_new = _energy_and_grad(X_new)
        
        # Half step velocity update
        V = V - 0.5 * dt * grad_new
        # Project velocity again
        V_dot_X = np.sum(V * X_new, axis=1)
        V = V - V_dot_X[:, np.newaxis] * X_new
        
        # Damping to help convergence
        V *= 0.999
        
        X_curr = X_new
        energy_curr = energy_new
        grad_curr = grad_new
        
        # Check convergence
        if np.linalg.norm(grad_curr) < tol:
            break
    
    return X_curr, energy_curr


# ---------------------------------------------------------------------------
# Gradient descent on the sphere with backtracking
# ---------------------------------------------------------------------------
def _gradient_descent_sphere(X, max_iter=500, tol=1e-10, deadline=None):
    """
    Riemannian gradient descent with backtracking line search.
    Returns (X_final, energy_final).
    """
    X_curr = X.copy()
    energy_curr, grad_curr = _energy_and_grad(X_curr)
    
    step_size = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Take a step using exponential map
        X_new = _exp_map_tangent(X_curr, -step_size * grad_curr)
        
        energy_new, grad_new = _energy_and_grad(X_new)
        
        # Check if energy decreased
        if energy_new < energy_curr:
            X_curr = X_new
            energy_curr = energy_new
            grad_curr = grad_new
            step_size = min(step_size * 1.5, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        
        # Check convergence
        if np.linalg.norm(grad_curr) < tol:
            break
    
    return X_curr, energy_curr


# ---------------------------------------------------------------------------
# High-precision polish pass
# ---------------------------------------------------------------------------
def _high_precision_polish(X, deadline=None):
    """
    Final high-precision polish pass with very tight tolerance.
    Combines Riemannian L-BFGS and Cartesian L-BFGS for robustness.
    """
    X_curr = X.copy()
    
    # First: Riemannian L-BFGS polish
    if deadline is not None and time.time() < deadline - 2.0:
        X_r, e_r = _riemannian_lbfgs_polish(X_curr, deadline=deadline - 2.0, maxiter=300, ftol=1e-18, gtol=1e-15)
        if e_r < _energy_only(X_curr):
            X_curr = X_r
    
    # Second: Cartesian L-BFGS polish as fallback/verification
    if deadline is not None and time.time() < deadline - 1.0:
        X_c, e_c = _lbfgs_polish(X_curr, deadline=deadline - 1.0, maxiter=200, ftol=1e-18, gtol=1e-15)
        if e_c < _energy_only(X_curr):
            X_curr = X_c
    
    return X_curr, _energy_only(X_curr)


# ---------------------------------------------------------------------------
# Parallel tempering (replica exchange) on the sphere
# ---------------------------------------------------------------------------
def _parallel_tempering(X_initials, temperatures, max_iter=200, deadline=None):
    """
    Parallel tempering (replica exchange) on the sphere.
    Multiple replicas at different temperatures explore the energy landscape.
    
    Parameters:
    - X_initials: list of (N,3) arrays, one per replica
    - temperatures: list of temperature values (higher = more exploration)
    - max_iter: number of iterations
    - deadline: time deadline
    
    Returns: list of (X, energy) tuples for each replica
    """
    n_replicas = len(X_initials)
    replicas = [X.copy() for X in X_initials]
    energies = []
    grads = []
    
    # Initialize each replica
    for i in range(n_replicas):
        e, g = _energy_and_grad(replicas[i])
        energies.append(e)
        grads.append(g)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Each replica takes a gradient step with temperature-dependent noise
        for i in range(n_replicas):
            T = temperatures[i]
            
            # Gradient descent step
            step_size = 0.005 * (1.0 + T)
            
            X_new = _exp_map_tangent(replicas[i], -step_size * grads[i])
            
            # Add thermal noise proportional to temperature
            if T > 0.01:
                noise_scale = 0.001 * T
                noise = noise_scale * np.random.randn(N, 3)
                X_new = _project_to_sphere(X_new + noise)
            
            # Accept with Metropolis criterion if energy increases
            e_new, g_new = _energy_and_grad(X_new)
            
            if e_new <= energies[i]:
                replicas[i] = X_new
                energies[i] = e_new
                grads[i] = g_new
            else:
                # Metropolis acceptance: accept with probability exp(-(e_new - e)/T)
                if T > 1e-6:
                    delta_e = e_new - energies[i]
                    if delta_e / T < 50:  # Avoid overflow
                        p_accept = math.exp(-delta_e / T)
                        if np.random.random() < p_accept:
                            replicas[i] = X_new
                            energies[i] = e_new
                            grads[i] = g_new
        
        # Replica exchange: try to swap adjacent replicas
        for i in range(n_replicas - 1):
            if temperatures[i] < temperatures[i + 1]:
                # Accept swap if it would lower the total "free energy"
                if energies[i] > energies[i + 1] and temperatures[i] < temperatures[i + 1]:
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    grads[i], grads[i + 1] = grads[i + 1], grads[i]
                elif np.random.random() < 0.3:
                    # Occasionally swap regardless to maintain ergodicity
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    grads[i], grads[i + 1] = grads[i + 1], grads[i]
    
    return list(zip(replicas, energies))


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_X)
    
    # Phase 1: Multi-start with gradient descent
    n_starts = 12
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            X = _fibonacci_spiral(N)
        elif start_idx == 1:
            X = _geodesic_dome(N, seed=42)
        elif start_idx < 6:
            # Perturbed Fibonacci spiral
            offset = np.random.uniform(-0.5, 0.5)
            X = _fibonacci_spiral(N, offset=offset)
            X += 0.01 * np.random.randn(N, 3)
            X = _project_to_sphere(X)
        else:
            # Random uniform on sphere
            X = _random_sphere(N, seed=start_idx * 100)
        
        # Gradient descent
        X_curr, energy_curr = _gradient_descent_sphere(X, max_iter=400, tol=1e-10, deadline=deadline - 40.0)
        
        # Velocity-Verlet dynamics for basin exploration
        if time.time() < deadline - 35.0:
            X_vv, energy_vv = _velocity_verlet_sphere(X_curr, dt=0.005, max_iter=80, tol=1e-8, deadline=deadline - 35.0)
            if energy_vv < energy_curr:
                X_curr, energy_curr = X_vv, energy_vv
                # Polish after VV
                X_curr, energy_curr = _gradient_descent_sphere(X_curr, max_iter=150, tol=1e-12, deadline=deadline - 35.0)
        
        # Update best
        if energy_curr < best_energy:
            best_energy = energy_curr
            best_X = X_curr.copy()
    
    # Phase 2: Parallel tempering to explore the landscape
    if time.time() < deadline - 20.0:
        # Create multiple replicas from the best solution and some perturbations
        n_replicas = 5
        X_initials = []
        temperatures = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        # Replica 0: best solution (cold, fine-tuning)
        X_initials.append(best_X.copy())
        
        # Replicas 1-4: increasingly perturbed versions
        for i in range(1, n_replicas):
            X_pert = best_X.copy()
            perturbation_scale = 0.005 * (i ** 1.5)
            X_pert += perturbation_scale * np.random.randn(N, 3)
            X_pert = _project_to_sphere(X_pert)
            X_initials.append(X_pert)
        
        # Run parallel tempering
        pt_results = _parallel_tempering(
            X_initials, 
            temperatures, 
            max_iter=150, 
            deadline=deadline - 15.0
        )
        
        # Find the best result from parallel tempering
        for X_pt, e_pt in pt_results:
            if e_pt < best_energy:
                # Polish this result
                X_polished, e_polished = _gradient_descent_sphere(
                    X_pt, max_iter=300, tol=1e-12, deadline=deadline - 10.0
                )
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_X = X_polished.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        X_polished, energy_polished = _high_precision_polish(best_X, deadline=deadline - 2.0)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_X = X_polished
    
    # Ensure all points are on the unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Check for any issues
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-9):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral(N)
    
    return best_X
# EVOLVE-BLOCK-END
