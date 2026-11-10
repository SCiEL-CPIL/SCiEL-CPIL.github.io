# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
import scipy.optimize as opt
from scipy.optimize import linprog
import warnings

warnings.filterwarnings('ignore')


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Uses a combination of SDP-inspired initialization (via known symmetric structures)
    and local optimization (L-BFGS-B) with multiple random restarts.
    
    Returns
        points: np.ndarray of shape (14,3)
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit
    
    n = 14
    d = 3
    
    def score(points):
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points, metric='euclidean')
        if dists.size == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(points):
        # We want to maximize (d_min/d_max)^2
        # Equivalent to minimizing -log(d_min/d_max) or similar
        # But d_min is non-smooth. 
        # Alternative: minimize (d_max / d_min)
        # Or use a smooth approximation.
        
        # Let's try minimizing the ratio d_max / d_min directly.
        # This is not differentiable at d_min, but L-BFGS can often handle it.
        # Better: minimize -log(d_min) + log(d_max) = log(d_max/d_min)
        # Even better for optimization: minimize d_max^2 / d_min^2
        
        dists = pdist(points, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_min < 1e-10:
            return 1e10
        if d_max < 1e-10:
            return 1e10
            
        # We want to minimize d_max / d_min
        # To avoid division by zero and improve conditioning, we can normalize
        # Let's minimize (d_max / d_min)
        return d_max / d_min

    def objective_grad(points):
        # Gradient of d_max / d_min is complex.
        # Let's use numerical gradient or just use L-BFGS with finite differences.
        # Given the small dimension (14*3=42), finite differences are okay.
        # But we can also provide a gradient if we know which pair is min/max.
        # For simplicity and robustness, let's rely on L-BFGS's finite difference
        # or provide a simple gradient.
        
        # Actually, let's just use L-BFGS-B without providing gradient initially.
        # If it's slow, we can add gradient later.
        pass

    # Strategy:
    # 1. Generate diverse initial points.
    #    - Random points
    #    - Points on a sphere (Fibonacci sphere or random)
    #    - Points from regular polyhedra if possible (14 is not a regular polyhedron vertex count)
    #    - Symmetric configurations (e.g., two tetrahedra, or octahedron + points)
    
    # 2. For each initial point, run L-BFGS-B to minimize d_max/d_min.
    
    # 3. Keep the best result.
    
    best_points = None
    best_score = -1.0
    best_ratio = 0.0
    
    # Generate initial points
    inits = []
    
    # 1. Random points
    np.random.seed(42)
    for i in range(20):
        pts = np.random.randn(n, d)
        # Normalize to unit sphere to start
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        pts /= norms
        inits.append(pts)
        
    # 2. Fibonacci sphere points
    def fibonacci_sphere(n_points):
        points = []
        phi = np.arccos(1 - 2 * (np.arange(n_points) + 0.5) / n_points)
        theta = np.pi * (1 + 5**0.5) * np.arange(n_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))
    
    fib_pts = fibonacci_sphere(n)
    inits.append(fib_pts)
    
    # 3. Random rotations of Fibonacci sphere
    for i in range(10):
        R = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(R)
        inits.append(fib_pts @ Q.T)
        
    # 4. Two tetrahedra (4+4+4+2? No, 14 points)
    # Let's try 4+4+4+2
    # Or just random clusters
    
    # 5. Points on cube + face centers + ...?
    # Cube has 8 vertices. Add 6 face centers. Total 14.
    # Let's try this structure.
    cube_vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    face_centers = np.array([
        [0, 0, 1], [0, 0, -1],
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0]
    ])
    cube_face_pts = np.vstack([cube_vertices, face_centers])
    # Normalize?
    norms = np.linalg.norm(cube_face_pts, axis=1, keepdims=True)
    cube_face_pts /= norms
    inits.append(cube_face_pts)
    
    # Random rotations of cube+face centers
    for i in range(10):
        R = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(R)
        inits.append(cube_face_pts @ Q.T)
        
    # 6. Octahedron (6 vertices) + Cube (8 vertices) = 14 points
    octahedron = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    cube_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    octa_cube_pts = np.vstack([octahedron, cube_verts])
    norms = np.linalg.norm(octa_cube_pts, axis=1, keepdims=True)
    octa_cube_pts /= norms
    inits.append(octa_cube_pts)
    
    for i in range(10):
        R = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(R)
        inits.append(octa_cube_pts @ Q.T)

    # Optimization loop
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        try:
            # L-BFGS-B optimization
            result = opt.minimize(
                objective,
                init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_pts = result.x.reshape(n, d)
            
            # Calculate score
            s = score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts
                dists = pdist(opt_pts, metric='euclidean')
                best_ratio = np.min(dists) / np.max(dists)
                
        except Exception as e:
            continue
            
    # If no optimization succeeded, return best initial point
    if best_points is None:
        best_points = inits[0]
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        return np.zeros((n, d))
        
    return best_points


# EVOLVE-BLOCK-END
