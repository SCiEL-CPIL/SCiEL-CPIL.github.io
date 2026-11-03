"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Returns (energy, grad) where grad is the Euclidean gradient (not yet projected).
    """
    n = len(pts)
    # Vectorized pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dists_sq, 1.0)  # avoid log(0)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # For each pair (i,j), contribution to grad[i] is -(pts[i]-pts[j])/d^2, to grad[j] is +(pts[i]-pts[j])/d^2
    # grad[i] = sum_{j != i} -(pts[i] - pts[j]) / |pts[i] - pts[j]|^2
    #         = sum_{j != i} (pts[j] - pts[i]) / dists_sq[i,j]
    
    # Build gradient matrix: G[i,j] = (pts[j] - pts[i]) / dists_sq[i,j] for i != j
    # grad[i] = sum_j G[i,j]
    
    # Vectorized: for each i, grad[i] = sum_j (pts[j] - pts[i]) / dists_sq[i,j]
    # = (sum_j pts[j] / dists_sq[i,j]) - pts[i] * sum_j 1/dists_sq[i,j]
    
    # inv_dists_sq[i,j] = 1/dists_sq[i,j] for i != j, 0 for i==j
    inv_dists_sq = np.zeros_like(dists_sq)
    np.fill_diagonal(inv_dists_sq, 0.0)
    inv_dists_sq = 1.0 / np.maximum(dists_sq, 1e-30)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # sum_j pts[j] / dists_sq[i,j]
    # This is: inv_dists_sq @ pts (matrix-vector product)
    term1 = inv_dists_sq @ pts  # shape (n, 3)
    
    # sum_j 1/dists_sq[i,j]
    term2 = inv_dists_sq.sum(axis=1, keepdims=True)  # shape (n, 1)
    
    grad = term1 - pts * term2
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _icosahedron_based(n, rng):
    """Generate points based on icosahedron subdivision + perturbation."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = _project_to_sphere(ico)
    
    # Subdivide faces and project to sphere, then perturb
    pts = _fibonacci_sphere(n).copy()
    pts = _project_to_sphere(pts + 0.02 * rng.randn(n, 3))
    return pts


def _lbfgs_on_sphere(pts, maxiter=2000, ftol=1e-14, gtol=1e-12, maxfun=5000, deadline=None):
    """Run L-BFGS-B optimization on the sphere using flat parameterization."""
    
    def objective_and_grad(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_tangent = g - np.einsum('ij,ij->i', g, p)[:, None] * p
        return e, g_tangent.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxfun}
        )
        if result.fun is not None and np.isfinite(result.fun):
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_e, _ = _energy_and_grad(new_pts)
            if np.isfinite(new_e):
                return new_pts, new_e
    except Exception:
        pass
    
    return pts, _energy_and_grad(pts)[0]


def _simulated_annealing(pts, rng, n_steps=5000, t_start=0.5, t_end=0.01, deadline=None):
    """Simulated annealing on the sphere to escape local minima."""
    current_pts = pts.copy()
    current_energy, _ = _energy_and_grad(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        i = rng.randint(N)
        
        # Generate a random perturbation in tangent space
        v = rng.randn(3)
        v -= np.dot(v, current_pts[i]) * current_pts[i]  # project to tangent
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            continue
        v = v / v_norm
        
        # Step size
        temperature = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
        step_size = 0.1 * temperature
        
        new_pts = current_pts.copy()
        new_pts[i] = _project_to_sphere(current_pts[i] + step_size * v)
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / max(temperature, 1e-10)):
            current_pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Best solution found so far
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N).copy()),
        ("fib_perturb1", _project_to_sphere(_fibonacci_sphere(N) + 0.01 * rng.randn(N, 3))),
        ("fib_perturb2", _project_to_sphere(_fibonacci_sphere(N) + 0.05 * rng.randn(N, 3))),
        ("random1", _random_sphere(N, rng)),
        ("random2", _random_sphere(N, rng)),
        ("random3", _random_sphere(N, rng)),
        ("ico_based", _icosahedron_based(N, rng)),
        ("fib_perturb3", _project_to_sphere(_fibonacci_sphere(N) + 0.1 * rng.randn(N, 3))),
    ]
    
    # Phase 1: Optimize each initialization with L-BFGS-B
    for name, init_pts in initializations:
        if time.time() > deadline - 15.0:
            break
        
        opt_pts, opt_energy = _lbfgs_on_sphere(
            init_pts,
            maxiter=1000,
            ftol=1e-13,
            gtol=1e-11,
            maxfun=3000,
            deadline=deadline - 10.0
        )
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Phase 2: Simulated annealing from best points to escape local minima
    if time.time() < deadline - 30.0:
        sa_pts, sa_energy = _simulated_annealing(
            best_pts, rng,
            n_steps=3000,
            t_start=0.3,
            t_end=0.005,
            deadline=deadline - 25.0
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts.copy()
    
    # Phase 3: Final L-BFGS polish from best points
    if time.time() < deadline - 20.0:
        opt_pts, opt_energy = _lbfgs_on_sphere(
            best_pts,
            maxiter=3000,
            ftol=1e-15,
            gtol=1e-13,
            maxfun=10000,
            deadline=deadline - 10.0
        )
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Phase 4: Another SA round from the polished result
    if time.time() < deadline - 25.0:
        sa_pts, sa_energy = _simulated_annealing(
            best_pts, rng,
            n_steps=2000,
            t_start=0.1,
            t_end=0.002,
            deadline=deadline - 20.0
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts.copy()
        
        # Final polish
        if time.time() < deadline - 10.0:
            opt_pts, opt_energy = _lbfgs_on_sphere(
                best_pts,
                maxiter=2000,
                ftol=1e-15,
                gtol=1e-13,
                maxfun=5000,
                deadline=deadline - 5.0
            )
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = dists_sq.min()
    
    if min_dist_sq < 1e-18:  # dist < 1e-9
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
