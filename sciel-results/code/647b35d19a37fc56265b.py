"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s eval / 150s kill

    def compute_covering_radius_and_worst(points):
        """Compute covering radius and identify worst facets."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
            worst_facets = []
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                if facet_radius > max_radius:
                    max_radius = facet_radius
                    worst_facets = [(simplex, circumcenter, facet_radius)]
                elif abs(facet_radius - max_radius) < 1e-8:
                    worst_facets.append((simplex, circumcenter, facet_radius))
            return max_radius, worst_facets
        except Exception:
            return np.pi, []

    def project_to_sphere(x):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return x / norms

    def riemannian_sgd_step(points, lr, momentum=0.9):
        """
        Riemannian SGD step focusing on worst facets.
        Moves vertices of worst facets towards their circumcenters.
        """
        radius, worst_facets = compute_covering_radius_and_worst(points)
        if not worst_facets:
            return points, radius, np.zeros_like(points)
        
        # Compute subgradient: push vertices toward circumcenter of worst facets
        subgrad = np.zeros_like(points)
        target_radius = radius
        for simplex, circumcenter, facet_r in worst_facets:
            # Weight by how much this facet exceeds target
            weight = 1.0  # All worst facets get equal weight
            for idx in simplex:
                vertex = points[idx]
                # Riemannian gradient: project tangent direction
                diff = circumcenter - vertex
                # Project to tangent space at vertex: diff - (diff·v)v
                dot_v = np.dot(diff, vertex)
                tangent = diff - dot_v * vertex
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-10:
                    subgrad[idx] += weight * tangent / tangent_norm
        
        # Normalize subgradient
        subgrad_norms = np.linalg.norm(subgrad, axis=1, keepdims=True)
        subgrad_norms = np.where(subgrad_norms < 1e-10, 1.0, subgrad_norms)
        subgrad = subgrad / subgrad_norms
        
        # Riemannian gradient descent step
        new_pts = points - lr * subgrad
        new_pts = project_to_sphere(new_pts)
        
        new_radius, _ = compute_covering_radius_and_worst(new_pts)
        return new_pts, new_radius, subgrad

    # Initialize with multiple diverse starting points
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_uniform(n):
        pts = np.random.randn(n, 3)
        return project_to_sphere(pts)

    # Best solution tracking
    best_pts = None
    best_radius = np.pi

    # Try multiple initializations
    initializations = [
        fibonacci_spiral(N),
        random_uniform(N),
        random_uniform(N),
    ]
    
    # Add perturbed Fibonacci
    fib = fibonacci_spiral(N)
    for _ in range(3):
        perturbed = fib + 0.1 * np.random.randn(N, 3)
        initializations.append(project_to_sphere(perturbed))

    for init_pts in initializations:
        if time.time() - start_time > time_limit - 30:
            break
        
        pts = init_pts.copy()
        current_radius, _ = compute_covering_radius_and_worst(pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        # Riemannian SGD with annealing
        lr = 0.05
        momentum = 0.9
        v = np.zeros_like(pts)  # Momentum buffer
        stagnant_count = 0
        
        for iteration in range(3000):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Adaptive learning rate with annealing
            lr = 0.05 * (0.995 ** iteration)
            
            # Riemannian SGD step with momentum
            new_pts, new_radius, subgrad = riemannian_sgd_step(pts, lr, momentum)
            
            # Update momentum
            v = momentum * v + (1 - momentum) * subgrad
            
            # Apply momentum
            new_pts = pts - lr * v
            new_pts = project_to_sphere(new_pts)
            
            # Recompute radius
            new_radius, _ = compute_covering_radius_and_worst(new_pts)
            
            if new_radius < current_radius - 1e-10:
                pts = new_pts
                current_radius = new_radius
                stagnant_count = 0
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            else:
                stagnant_count += 1
                if stagnant_count > 100:
                    # Restart with perturbation
                    pts = pts + 0.05 * np.random.randn(N, 3)
                    pts = project_to_sphere(pts)
                    current_radius, _ = compute_covering_radius_and_worst(pts)
                    v = np.zeros_like(pts)
                    stagnant_count = 0
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()
        
        # Fine polish on best result
        if best_pts is not None:
            pts = best_pts.copy()
            current_radius = best_radius
            lr = 0.01
            for iteration in range(500):
                if time.time() - start_time > time_limit - 2:
                    break
                if current_radius < 0.1972:
                    break
                
                new_pts, new_radius, _ = riemannian_sgd_step(pts, lr)
                
                if new_radius < current_radius - 1e-12:
                    pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()
                    lr *= 0.998
                else:
                    lr *= 0.95
                    if lr < 1e-5:
                        break

    if best_pts is None:
        best_pts = fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
