"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math


def construct():
    """
    Construct 150 points on the unit sphere to minimize logarithmic energy.
    Strategy:
    1. Generate initial points via subdivided icosahedron (geodesic dome) for uniform spacing.
    2. Refine using gradient descent on the sphere's tangent space.
    3. Use natural logarithm of chordal distances.
    """
    
    def icosahedron_vertices():
        """Return the 12 vertices of a regular icosahedron."""
        t = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [-1,  t,  0],
            [ 1,  t,  0],
            [-1, -t,  0],
            [ 1, -t,  0],
            [ 0, -1,  t],
            [ 0,  1,  t],
            [ 0, -1, -t],
            [ 0,  1, -t],
            [ t,  0, -1],
            [ t,  0,  1],
            [-t,  0, -1],
            [-t,  0,  1]
        ], dtype=float)
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        return verts
    
    def subdivide_faces(vertices, faces, subdivisions=1):
        """
        Subdivide each triangular face by inserting midpoints and projecting to sphere.
        Returns new vertices and faces.
        """
        # Cache for midpoints
        mid_cache = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in mid_cache:
                return mid_cache[key]
            v1 = vertices[i]
            v2 = vertices[j]
            mid = (v1 + v2) / 2.0
            mid = mid / np.linalg.norm(mid)
            new_idx = len(vertices)
            vertices = np.vstack([vertices, mid])
            mid_cache[key] = new_idx
            return new_idx, vertices
        
        new_faces = []
        for face in faces:
            a, b, c = face
            ab = get_midpoint(a, b)[0]
            bc = get_midpoint(b, c)[0]
            ca = get_midpoint(c, a)[0]
            # Update vertices if get_midpoint added new ones
            # This approach is tricky; let's use a different strategy
            pass
        
        # Simpler approach: use a known subdivision routine
        # For now, let's use a different initialization method
        return vertices, faces
    
    # Use a simpler high-quality initialization: 
    # Start with icosahedron and subdivide, or use a well-known point set
    
    # Alternative: Use the Fibonacci sphere but with a twist, or use a precomputed good set
    # Let's use a more systematic approach: start with icosahedron, subdivide once (42 verts),
    # then add more points or use a different method
    
    # Actually, let's use a different strategy: 
    # 1. Generate points using a well-distributed method (e.g., Leja points or similar)
    # 2. Optimize with gradient descent
    
    # Generate initial points using a spiral with some jitter, then optimize
    # Or use a known good starting point
    
    # Let's use a combination: start with a subdivided icosahedron
    # For 150 points, we can use an icosahedron subdivided to have approximately 150 vertices
    
    # Icosahedron has 12 vertices, 20 faces
    # After 1 subdivision: 12 + 20*3 = 72 vertices, 80 faces
    # After 2 subdivisions: 72 + 80*3 = 312 vertices, 320 faces
    # So 150 is between 1 and 2 subdivisions
    
    # Let's use a different approach: use a high-quality initial distribution
    # and then optimize
    
    # Initialize with a well-distributed set
    # Use the "sunflower" or "Fibonacci" sphere as initial guess, then optimize
    # But we want a stronger algorithm, so let's do proper optimization
    
    # Step 1: Initial points - use a modified Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Step 2: Gradient descent on the sphere
    # Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # Gradient of E with respect to x_i:
    # dE/dx_i = -sum_{j != i} (-1) * (x_i - x_j) / |x_i - x_j|^2 = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Wait, let's recompute:
    # E = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = -sum_{j != i} (1/2) * d(log|x_i - x_j|^2)/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # So gradient is -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # To minimize E, we move in the direction of -gradient = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    def compute_gradient(points):
        """Compute the gradient of the logarithmic energy."""
        n = len(points)
        grad = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-20:
                    continue
                grad[i] += diff / dist_sq
        return grad
    
    def project_to_sphere(points):
        """Project points back to the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return points / norms
    
    # Gradient descent with adaptive step size
    lr = 0.01
    n_iter = 5000
    points = points.copy()
    
    for it in range(n_iter):
        grad = compute_gradient(points)
        # Move in the direction of -gradient (since we want to minimize)
        # But the gradient formula above gives dE/dx_i, so to minimize we go in -grad direction
        points = points - lr * grad
        # Project back to sphere
        points = project_to_sphere(points)
        
        # Decay learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.9
    
    # Final normalization
    points = project_to_sphere(points)
    
    return points

# EVOLVE-BLOCK-END
