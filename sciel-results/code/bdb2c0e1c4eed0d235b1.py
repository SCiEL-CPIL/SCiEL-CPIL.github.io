import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _halton_sphere(n, base):
    """Generate points using Halton sequence on sphere."""
    def halton(index, b):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= b
            r += f * (index % b)
            index //= b
        return r
    
    pts = []
    for i in range(1, n + 1):
        u = halton(i, base)
        v = halton(i, base + 1)
        theta = 2.0 * np.pi * u
        phi = np.arccos(1.0 - 2.0 * v)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts)

def _icosahedron_subdivide(n_target):
    """
    Generate points by subdividing an icosahedron.
    This creates a symmetric starting point.
    """
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    vertices = _normalize(vertices)
    
    # Icosahedron faces (12 vertices, 20 faces)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces recursively until we have enough points
    # Each subdivision of a face creates 4 faces, increasing point count
    # We'll do a simple recursive subdivision and collect unique midpoints
    
    points = list(vertices)
    point_set = set()
    for p in points:
        point_set.add((round(p[0], 10), round(p[1], 10), round(p[2], 10)))
    
    # Subdivide each face
    new_faces = []
    for face in faces:
        a, b, c = [points[i] for i in face]
        # Midpoints
        ab = (a + b) / 2.0
        bc = (b + c) / 2.0
        ca = (c + a) / 2.0
        ab, bc, ca = _normalize(ab.reshape(1,3))[0], _normalize(bc.reshape(1,3))[0], _normalize(ca.reshape(1,3))[0]
        
        # Check if midpoints are new
        for mid in [ab, bc, ca]:
            key = (round(mid[0], 10), round(mid[1], 10), round(mid[2], 10))
            if key not in point_set:
                point_set.add(key)
                points.append(mid)
        
        new_faces.extend([
            [a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]
        ])
    
    # We now have 12 + 30 = 42 points
    # Continue subdividing if needed
    current_faces = new_faces
    while len(points) < n_target and len(points) < 200:
        next_faces = []
        for face in current_faces:
            a, b, c = [points[i] for i in face]
            ab = _normalize(((a + b) / 2.0).reshape(1,3))[0]
            bc = _normalize(((b + c) / 2.0).reshape(1,3))[0]
            ca = _normalize(((c + a) / 2.0).reshape(1,3))[0]
            
            for mid in [ab, bc, ca]:
                key = (round(mid[0], 10), round(mid[1], 10), round(mid[2], 10))
                if key not in point_set:
                    point_set.add(key)
                    points.append(mid)
            
            next_faces.extend([
                [a, points.index(ab) if ab in points else -1, points.index(ca) if ca in points else -1],
                [points.index(ab) if ab in points else -1, b, points.index(bc) if bc in points else -1],
                [points.index(ca) if ca in points else -1, points.index(bc) if bc in points else -1, c],
                [points.index(ab) if ab in points else -1, points.index(bc) if bc in points else -1, points.index(ca) if ca in points else -1]
            ])
        
        if len(next_faces) == 0:
            break
        current_faces = next_faces
    
    # If we have more than n_target, just take the first n_target
    # Or better: select n_target points that are most evenly distributed
    if len(points) >= n_target:
        # Select points using farthest-point sampling from the subdivided set
        selected = [0]  # Start with first point
        min_dists = np.array([np.linalg.norm(points[0] - p) for p in points])
        
        for _ in range(1, n_target):
            max_idx = np.argmax(min_dists)
            selected.append(max_idx)
            new_dists = np.linalg.norm(points[max_idx] - np.array(points), axis=1)
            min_dists = np.minimum(min_dists, new_dists)
        
        return np.array([points[i] for i in selected])
    else:
        # Pad with random points if needed
        result = np.array(points)
        if len(result) < n_target:
            remaining = n_target - len(result)
            random_pts = np.random.randn(remaining, 3)
            random_pts = _normalize(random_pts)
            result = np.vstack([result, random_pts])
        return result

def _energy_and_grad(pts):
    """
    Compute log energy and gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = np.sum(np.log(1.0 / dist[mask]))
    
    # forces[i] = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    forces = np.sum(-diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot_products = np.sum(forces * pts, axis=1)
    proj_forces = forces - dot_products[:, np.newaxis] * pts
    
    return energy, proj_forces

def _exponential_map(pts, tangents):
    """
    Exponential map on the sphere: move points along tangent vectors.
    """
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    angles = norms
    # Rodrigues' formula
    # For unit vector v and angle theta: exp(v) = cos(theta) * x + sin(theta) * (v/|v|)
    # Actually: x_new = cos(theta) * x + sin(theta) * v_hat where theta = |v|
    sin_theta = np.sin(angles) / norms
    cos_theta = np.cos(angles)
    new_pts = cos_theta * pts + sin_theta * tangents
    return _normalize(new_pts)

def _riemannian_lbfgs(pts, time_limit=30.0):
    """
    Riemannian L-BFGS optimization on the sphere.
    """
    start_time = time.time()
    
    # We'll use the 3D embedding with projection
    # State is 3N coordinates, but we project to sphere after each step
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return float('inf')
        p = _normalize(x.reshape(N, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient_func(x):
        if time.time() - start_time > time_limit:
            return np.zeros_like(x)
        p = _normalize(x.reshape(N, 3))
        _, g = _energy_and_grad(p)
        # The gradient in the ambient space needs to account for the constraint
        # For simplicity, we use the projected gradient as the ambient gradient
        # This is an approximation but works well in practice
        return g.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient_func,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.fun < float('inf'):
            res_pts = _normalize(result.x.reshape(N, 3))
            return res_pts, result.fun
    except Exception:
        pass
    
    return pts, float('inf')

def _riemannian_gd(pts, max_iters=3000, lr_init=0.02, lr_min=1e-7, time_limit=25.0):
    """
    Riemannian gradient descent with adaptive step size.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    lr = lr_init
    momentum = 0.8
    velocity = np.zeros_like(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Momentum update
        velocity = momentum * velocity + grad
        update = velocity
        
        # Adaptive step size with backtracking
        step = lr
        for _ in range(15):
            new_pts = _exponential_map(pts, -step * update)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            step *= 0.5
        
        new_pts = _exponential_map(pts, -step * update)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.6
            if lr < lr_min:
                break
    
    return best_pts, best_energy

def _polish(pts, time_limit=10.0):
    """
    Final polish using Newton-type steps with approximate Hessian.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Use L-BFGS with tighter tolerances
    lbfgs_pts, lbfgs_energy = _riemannian_lbfgs(best_pts, time_limit=time_limit)
    if lbfgs_energy < best_energy:
        best_energy = lbfgs_energy
        best_pts = lbfgs_pts
    
    return best_pts, best_energy

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Try multiple initializations with symmetric structure
    initializations = [
        _icosahedron_subdivide(N),
        _fibonacci_sphere(N),
        _halton_sphere(N, 2),
        _halton_sphere(N, 3),
    ]
    
    best_pts = None
    best_energy = float('inf')
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 35.0:
            break
        
        # Phase 1: Riemannian gradient descent for coarse optimization
        gd_pts, gd_energy = _riemannian_gd(
            init_pts,
            max_iters=2000,
            lr_init=0.02,
            lr_min=1e-7,
            time_limit=20.0
        )
        
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_pts = gd_pts.copy()
        
        # Phase 2: L-BFGS polish
        if time.time() > deadline - 20.0:
            break
        
        lbfgs_pts, lbfgs_energy = _riemannian_lbfgs(
            gd_pts,
            time_limit=15.0
        )
        
        if lbfgs_energy < best_energy:
            best_energy = lbfgs_energy
            best_pts = lbfgs_pts.copy()
        
        # Phase 3: Final polish
        if time.time() > deadline - 10.0:
            break
        
        final_pts, final_energy = _polish(best_pts, time_limit=8.0)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    # If we still have time, do one more round of polish on the best
    if best_pts is not None and time.time() < deadline - 5.0:
        final_pts, final_energy = _polish(best_pts, time_limit=5.0)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final verification
    final_pts = _normalize(best_pts)
    return final_pts
