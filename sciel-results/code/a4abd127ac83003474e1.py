import time
import numpy as np
from scipy.spatial.distance import pdist

DEADLINE = 620.0

def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 0.0
    if d_min < 1e-10:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _spherical_to_cartesian(sph):
    """Convert spherical coordinates (theta, phi per point) to Cartesian on unit sphere."""
    theta = sph[0::2]  # polar angle
    phi = sph[1::2]    # azimuthal angle
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _score_spherical(sph):
    """Score function in spherical coordinates."""
    pts = _spherical_to_cartesian(sph)
    return _raw_score(pts)

def _repulsion_simulate(theta, phi, n_iter=500, learning_rate=0.01):
    """
    Simple repulsion-based simulation to spread points on sphere.
    Points repel each other with force proportional to 1/d^2.
    """
    theta = theta.copy()
    phi = phi.copy()
    
    for it in range(n_iter):
        pts = _spherical_to_cartesian(np.column_stack([theta, phi]).flatten())
        
        # Compute pairwise forces
        forces = np.zeros((14, 3))
        for i in range(14):
            for j in range(i+1, 14):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d > 1e-10:
                    # Force magnitude ~ 1/d^2, direction along d_vec
                    f_mag = 1.0 / (d * d)
                    f_vec = d_vec / d * f_mag
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # Update positions (project back to sphere)
        for i in range(14):
            pts[i] += learning_rate * forces[i]
            norm = np.linalg.norm(pts[i])
            if norm > 1e-10:
                pts[i] = pts[i] / norm
        
        # Convert back to spherical
        z = pts[:, 2]
        theta = np.arccos(np.clip(z, -1, 1))
        xy = np.sqrt(np.maximum(0, 1 - z*z))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        phi = np.mod(phi, 2*np.pi)
    
    return theta, phi

def _nelder_mead_polish(theta, phi, max_iter=2000, timeout=None):
    """
    Nelder-Mead optimization on spherical coordinates to maximize (d_min/d_max)^2.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def neg_score(sph_flat):
        if timeout is not None and time.time() > timeout:
            return 1e10
        pts = _spherical_to_cartesian(sph_flat)
        score = _raw_score(pts)
        return -score
    
    x0 = np.column_stack([theta, phi]).flatten()
    
    try:
        res = minimize(
            neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        x_opt = res.x
        theta_opt = x_opt[0::2]
        phi_opt = x_opt[1::2]
        # Normalize angles
        theta_opt = np.clip(theta_opt, 0, np.pi)
        phi_opt = np.mod(phi_opt, 2*np.pi)
        return theta_opt, phi_opt
    except Exception:
        return theta, phi

def _generate_icosahedron():
    phi_val = (1 + np.sqrt(5)) / 2
    points = np.array([
        [0, 1, phi_val], [0, -1, phi_val], [0, 1, -phi_val], [0, -1, -phi_val],
        [1, phi_val, 0], [-1, phi_val, 0], [1, -phi_val, 0], [-1, -phi_val, 0],
        [phi_val, 0, 1], [phi_val, 0, -1], [-phi_val, 0, 1], [-phi_val, 0, -1]
    ])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return points

def _generate_fibonacci_sphere(n=14):
    """
    Generate n points on a sphere using the Fibonacci lattice.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        radius = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _cartesian_to_spherical(pts):
    """Convert Cartesian points on unit sphere to (theta, phi) pairs."""
    z = pts[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    xy = np.sqrt(np.maximum(0, 1 - z*z))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    phi = np.mod(phi, 2*np.pi)
    return theta, phi

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_points = np.zeros((14, 3))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Fibonacci sphere
    fib = _generate_fibonacci_sphere(14)
    initial_configs.append(fib)
    
    # 2. Icosahedron + 2 extra points
    ico = _generate_icosahedron()
    p1 = rng.normal(size=3)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=3)
    p2 = p2 / np.linalg.norm(p2)
    initial_configs.append(np.vstack([ico, p1, p2]))
    
    # 3. Cubic points
    cube = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ])
    norms = np.linalg.norm(cube, axis=1, keepdims=True)
    cube = cube / norms
    initial_configs.append(cube)
    
    # 4. Random points
    for _ in range(10):
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)
    
    # Optimize each configuration
    for config in initial_configs:
        if time.time() > deadline - 30:
            break
        
        # Convert to spherical
        theta, phi = _cartesian_to_spherical(config)
        
        # Phase 1: Repulsion simulation to spread points
        if time.time() < deadline - 60:
            theta, phi = _repulsion_simulate(theta, phi, n_iter=300, learning_rate=0.005)
        
        # Phase 2: Nelder-Mead polish
        if time.time() < deadline - 20:
            theta, phi = _nelder_mead_polish(theta, phi, max_iter=1500, timeout=deadline - 15)
        
        # Convert back to Cartesian and evaluate
        pts = _spherical_to_cartesian(np.column_stack([theta, phi]).flatten())
        score = _raw_score(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Additional random restarts if time permits
    while time.time() < deadline - 45:
        pts = rng.normal(size=(14, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        theta, phi = _cartesian_to_spherical(pts)
        theta, phi = _repulsion_simulate(theta, phi, n_iter=200, learning_rate=0.005)
        theta, phi = _nelder_mead_polish(theta, phi, max_iter=1000, timeout=deadline - 10)
        
        pts_final = _spherical_to_cartesian(np.column_stack([theta, phi]).flatten())
        score = _raw_score(pts_final)
        
        if score > best_score:
            best_score = score
            best_points = pts_final
    
    # Final polish on best points
    if time.time() < deadline - 10 and best_score > 0:
        theta, phi = _cartesian_to_spherical(best_points)
        theta, phi = _nelder_mead_polish(theta, phi, max_iter=2000, timeout=deadline - 5)
        pts_final = _spherical_to_cartesian(np.column_stack([theta, phi]).flatten())
        score = _raw_score(pts_final)
        if score > best_score:
            best_score = score
            best_points = pts_final
    
    if best_score == 0.0:
        # Fallback
        best_points = _generate_fibonacci_sphere(14)
    
    return best_points

if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    print(_raw_score(pts))
