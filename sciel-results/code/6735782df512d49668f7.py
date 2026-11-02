"""Combined logarithmic energy minimization for 100 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|).
    Uses math.fsum for accuracy as specified in the task."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    terms = [-math.log(d) for d in d_ij]
    return math.fsum(terms)

def _log_energy_fast(pts):
    """Vectorized energy for fast line search (slightly less accurate but much faster)."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    return -np.sum(np.log(d_ij))

def _gradient_tangent(pts):
    """Tangent-space gradient of logarithmic energy on the sphere."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)  # (n,)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T  # (n, 3)
    
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)  # (n, 3)
    
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)  # (n, 1)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _riemannian_hessian(pts):
    """Compute Riemannian Hessian of the energy. Returns a list of (i, j, H_ij) where H_ij is 3x3.
    For efficiency, we compute it in tangent space.
    E = -sum_{i<j} log(d_ij)
    d2E/dx_i dx_j for i != j: 2*(x_i-x_j)/d^4 - 2*(x_i-x_j)(x_i-x_j)^T/d^6
    d2E/dx_i dx_i: sum_{j!=i} [2/d^2 - 2*(x_i-x_j)^2/d^4] (diagonal in full space)
    Then project to tangent space.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    d = np.sqrt(d2)
    inv_d2 = 1.0 / d2
    inv_d4 = inv_d2 * inv_d2
    inv_d6 = inv_d4 * inv_d2
    np.fill_diagonal(inv_d2, 0.0)
    np.fill_diagonal(inv_d4, 0.0)
    np.fill_diagonal(inv_d6, 0.0)
    
    # Hessian in ambient space: H[i,j] for i != j is:
    # H_ij = 2*(x_i - x_j)/d^4 - 2*(x_i-x_j)(x_i-x_j)^T/d^6
    # For i == j: H_ii = sum_{k!=i} [2/d_ik^2 * I - 2*(x_i-x_k)(x_i-x_k)^T/d_ik^4]
    
    # We'll compute the full ambient Hessian as a 3n x 3n matrix
    # Then project to tangent space
    H_ambient = np.zeros((3*n, 3*n))
    
    for i in range(n):
        for j in range(i+1, n):
            diff = pts[i] - pts[j]  # (3,)
            d2_ij = d2[i, j]
            d4_ij = d2_ij * d2_ij
            d6_ij = d4_ij * d2_ij
            
            # H_ij (off-diagonal block)
            h_ij = 2.0 * diff / d4_ij - 2.0 * np.outer(diff, diff) / d6_ij
            H_ambient[3*i:3*i+3, 3*j:3*j+3] = h_ij
            H_ambient[3*j:3*j+3, 3*i:3*i+3] = h_ij
    
    # Diagonal blocks
    for i in range(n):
        H_ii = np.zeros((3, 3))
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d2_ij = d2[i, j]
            d4_ij = d2_ij * d2_ij
            H_ii += 2.0 / d2_ij * np.eye(3) - 2.0 * np.outer(diff, diff) / d4_ij
        H_ambient[3*i:3*i+3, 3*i:3*i+3] = H_ii
    
    # Project to tangent space: for each point, P_i = I - x_i x_i^T
    # H_tangent[i,j] = P_i @ H_ambient[i,j] @ P_j
    H_tangent = np.zeros((3*n, 3*n))
    for i in range(n):
        P_i = np.eye(3) - np.outer(pts[i], pts[i])
        for j in range(n):
            P_j = np.eye(3) - np.outer(pts[j], pts[j])
            H_tangent[3*i:3*i+3, 3*j:3*j+3] = P_i @ H_ambient[3*i:3*i+3, 3*j:3*j+3] @ P_j
    
    return H_tangent

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_subdivide(n_target, seed=42):
    """Generate points by subdividing an icosahedron, then selecting n_target points."""
    rng = np.random.RandomState(seed)
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide faces
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ]
    
    # Subdivide each face into 4 smaller triangles
    new_verts = [verts]
    new_faces = []
    for _ in range(3):  # 3 levels of subdivision: 20 -> 80 -> 320 -> 1280
        new_faces = []
        edge_midpoints = {}
        for f in new_faces:
            a, b, c = f
            # Find or create midpoints
            def get_midpoint(p1, p2):
                key = (min(p1, p2), max(p1, p2))
                if key not in edge_midpoints:
                    mid = (verts[p1] + verts[p2]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints[key] = len(verts)
                    verts = np.vstack([verts, mid.reshape(1, 3)])
                return edge_midpoints[key]
            
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        new_faces = new_faces
    
    # Now we have len(verts) points. Select n_target of them.
    n_available = len(verts)
    if n_available <= n_target:
        return np.array(verts)
    
    # Select n_target points evenly
    indices = np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.array(verts[indices])

def _random_points(n, seed=42):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _perturbed_points(base_pts, seed=0, sigma=0.03):
    """Generate perturbed version of base points."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    return _project_to_sphere(base_pts + noise)

def _lbfgs_tangent(pts, max_iter=2000, ftol=1e-12, gtol=1e-10, maxls=20, deadline=None):
    """L-BFGS-B optimization using tangent-space gradient."""
    x0 = pts.flatten()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e = _log_energy(p)
        g = _gradient_tangent(p)
        return e, g.flatten()
    
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _log_energy(best_pts)
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': maxls})
        if np.isfinite(res.fun) and res.fun < best_energy:
            new_pts = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(new_pts)
            if e_new < best_energy:
                best_pts = new_pts
                best_energy = e_new
    except Exception:
        pass
    
    return best_pts, best_energy

def _riemannian_cg(pts, max_iter=500, tol=1e-12, deadline=None):
    """Riemannian conjugate gradient method on the sphere."""
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    grad = _gradient_tangent(pts)
    p = -grad  # initial search direction
    grad_norm = np.linalg.norm(grad)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if grad_norm < tol:
            break
        
        # Line search
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        accepted = False
        for ls_iter in range(40):
            new_pts = _project_to_sphere(pts + step * p)
            new_f = _log_energy_fast(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-8 * step * np.linalg.norm(p) * np.linalg.norm(p) or new_f < f - 1e-10:
                    accepted = True
                    break
            step *= 0.7
        
        if not accepted:
            break
        
        new_grad = _gradient_tangent(new_pts)
        new_grad_norm = np.linalg.norm(new_grad)
        
        # Polak-Ribiere beta
        beta = max(0.0, np.sum(new_grad * (new_grad + grad)) / max(np.sum(grad * grad), 1e-15))
        
        pts = new_pts
        f = _log_energy(pts)  # Use accurate energy for tracking
        grad = new_grad
        grad_norm = new_grad_norm
        p = -grad + beta * p
        
    return pts, f

def _newton_polish(pts, max_iter=20, tol=1e-14, deadline=None):
    """Newton method using Riemannian Hessian for final polish."""
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_tangent(pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        try:
            H = _riemannian_hessian(pts)
            # Solve H * delta = -grad in tangent space
            # Use damped Newton for stability
            grad_flat = grad.flatten()
            
            # Add damping for positive definiteness
            delta = np.linalg.solve(H + 1e-6 * np.eye(3*N), -grad_flat)
            delta = delta.reshape(N, 3)
            
            # Project delta to tangent space
            for i in range(N):
                delta[i] -= np.dot(delta[i], pts[i]) * pts[i]
            
            # Line search with damped Newton step
            step = 1.0
            accepted = False
            for ls_iter in range(30):
                new_pts = _project_to_sphere(pts + step * delta)
                new_f = _log_energy_fast(new_pts)
                
                if np.isfinite(new_f) and new_f < f:
                    accepted = True
                    break
                step *= 0.5
            
            if not accepted:
                break
            
            pts = new_pts
            f = _log_energy(pts)
            
        except Exception:
            break
    
    return pts, f

def _riemannian_descent(pts, max_iter=1000, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_tangent(pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        step = 1.0 / (1.0 + 0.3 * grad_norm)
        
        accepted = False
        for ls_iter in range(40):
            new_pts = _project_to_sphere(pts - step * grad)
            new_f = _log_energy_fast(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-10 * step * grad_norm * grad_norm or new_f < f - 1e-12:
                    accepted = True
                    break
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = _log_energy(pts)
        else:
            break
    
    return pts, f

def _optimize_pipeline(pts0, deadline):
    """Multi-stage optimization: L-BFGS -> Riemannian CG -> Newton polish."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    # Stage 1: L-BFGS with moderate tolerance
    if time.time() < deadline - 5.0:
        pts, e = _lbfgs_tangent(best_pts, max_iter=1500, ftol=1e-12, gtol=1e-10, maxls=20, deadline=deadline - 5.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Stage 2: Riemannian conjugate gradient
    if time.time() < deadline - 3.0:
        pts, e = _riemannian_cg(best_pts, max_iter=300, tol=1e-12, deadline=deadline - 3.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Stage 3: Tight L-BFGS polish
    if time.time() < deadline - 2.0:
        pts, e = _lbfgs_tangent(best_pts, max_iter=2000, ftol=1e-14, gtol=1e-12, maxls=30, deadline=deadline - 2.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Stage 4: Newton polish
    if time.time() < deadline - 1.0:
        pts, e = _newton_polish(best_pts, max_iter=15, tol=1e-14, deadline=deadline - 1.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s
    
    # Start with multiple diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # Icosahedron subdivision
    try:
        initializations.append(_icosahedron_subdivide(N, seed=42))
    except Exception:
        pass
    
    # Random starts
    for seed in [1, 7, 42, 123, 789]:
        initializations.append(_random_points(N, seed=seed))
    
    # Perturbed Fibonacci
    fib = _fibonacci_spiral(N)
    for seed, sigma in [(10, 0.05), (20, 0.1), (30, 0.15)]:
        initializations.append(_perturbed_points(fib, seed=seed, sigma=sigma))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Optimize each initialization with time budget
    n_init = len(initializations)
    per_init_budget = 120.0 / n_init
    
    for idx, pts_init in enumerate(initializations):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time: more for first few, less for later
        alloc = min(per_init_budget, remaining - 10.0)
        if alloc < 3.0:
            alloc = remaining - 10.0
            if alloc < 2.0:
                break
        
        pts, e = _optimize_pipeline(pts_init, time.time() + alloc)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Final high-precision polish on the best configuration
    if time.time() < deadline - 3.0:
        pts, e = _lbfgs_tangent(best_pts, max_iter=5000, ftol=1e-15, gtol=1e-13, maxls=40, deadline=deadline - 3.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Newton polish
    if time.time() < deadline - 1.5:
        pts, e = _newton_polish(best_pts, max_iter=25, tol=1e-15, deadline=deadline - 1.5)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Final Riemannian CG pass
    if time.time() < deadline - 1.0:
        pts, e = _riemannian_cg(best_pts, max_iter=100, tol=1e-14, deadline=deadline - 1.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
