"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _covering_radius(points):
    """Compute the covering radius (max angular distance to nearest point) in degrees.
    
    Uses convex hull facets: the covering radius is the maximum circumradius of 
    spherical triangles (facets of the convex hull).
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use nearest-neighbor based estimate
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample many points on sphere
        n_samples = 10000
        theta = np.random.uniform(0, np.pi, n_samples)
        phi = np.random.uniform(0, 2 * np.pi, n_samples)
        samples = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        dists, _ = tree.query(samples)
        return np.degrees(np.arccos(np.clip(dists, -1, 1))).max()
    
    # Get facets (simplices) of the convex hull
    # Each facet is a set of indices
    facets = hull.simplices
    
    max_circumradius = 0.0
    for facet in facets:
        if len(facet) == 3:
            # Triangle: compute circumradius on sphere
            a, b, c = points[facet[0]], points[facet[1]], points[facet[2]]
            # Compute circumcenter of spherical triangle
            # Using the formula for circumradius of a spherical triangle
            # The circumradius R satisfies: the circumcenter is equidistant from all 3 vertices
            # For a spherical triangle, circumradius can be computed as:
            # cos(R) = (cos(a) + cos(b) + cos(c) + 1) / (4 * cos(A/2)*cos(B/2)*cos(C/2))
            # But simpler: find the point equidistant from a, b, c on the sphere
            
            # Compute pairwise distances
            cos_ab = np.clip(np.dot(a, b), -1, 1)
            cos_bc = np.clip(np.dot(b, c), -1, 1)
            cos_ca = np.clip(np.dot(c, a), -1, 1)
            d_ab = np.arccos(cos_ab)
            d_bc = np.arccos(cos_bc)
            d_ca = np.arccos(cos_ca)
            
            # Spherical law of cosines to find angles
            # Angle A at vertex a
            cos_A = (cos_bc - cos_ab * cos_ca) / (np.sqrt(1 - cos_ab**2) * np.sqrt(1 - cos_ca**2) + 1e-15)
            cos_A = np.clip(cos_A, -1, 1)
            A = np.arccos(cos_A)
            
            cos_B = (cos_ca - cos_ab * cos_bc) / (np.sqrt(1 - cos_ab**2) * np.sqrt(1 - cos_bc**2) + 1e-15)
            cos_B = np.clip(cos_B, -1, 1)
            B = np.arccos(cos_B)
            
            cos_C = (cos_ab - cos_bc * cos_ca) / (np.sqrt(1 - cos_bc**2) * np.sqrt(1 - cos_ca**2) + 1e-15)
            cos_C = np.clip(cos_C, -1, 1)
            C = np.arccos(cos_C)
            
            # Circumradius: sin(R) = sin(a) / (2 * sin(A)) where a is side opposite A
            # Using: sin(R) = sin(d_bc) / (2 * sin(A))
            sin_R = np.sin(d_bc) / (2 * np.sin(A) + 1e-15)
            sin_R = np.clip(sin_R, 0, 1)
            R = np.arcsin(sin_R)
            
            if R > max_circumradius:
                max_circumradius = R
        elif len(facet) == 4:
            # Tetrahedron facet (shouldn't happen for convex hull in 3D, but handle)
            # For safety, just use max pairwise
            for i in range(len(facet)):
                for j in range(i+1, len(facet)):
                    d = np.arccos(np.clip(np.dot(points[facet[i]], points[facet[j]]), -1, 1))
                    if d > max_circumradius:
                        max_circumradius = d
    
    return np.degrees(max_circumradius)


def _objective_and_gradient(points, weights=None):
    """Objective: maximize minimum distance (or minimize max covering radius).
    
    We use a potential-energy-like objective: sum of -1/d^p for all pairs,
    which encourages uniform distribution.
    """
    if weights is None:
        weights = np.ones(N)
    
    points = _normalize(points)
    # Compute all pairwise distances
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots)
    
    # Potential energy: sum of -k/d^p
    p = 4
    k = 1.0
    np.fill_diagonal(dists, np.inf)
    energy = -np.sum(weights[:, None] * weights[None, :] * k / dists**p)
    
    # Gradient with respect to each point
    # d(energy)/d(points[i]) = sum_j weights[i]*weights[j] * p * k * (points[i] - (points[i]·points[j])*points[j]) / (1 - (points[i]·points[j])^2) * 1/dists[i,j]^p
    # This is complex; use finite differences or a simpler approach
    
    return energy


def _repulsion_refine(points, n_iters=500, dt=0.01):
    """Velocity Verlet integration with repulsive forces to spread points uniformly."""
    points = _normalize(points.copy())
    velocities = np.zeros_like(points)
    
    best_energy = np.inf
    best_points = points.copy()
    
    for it in range(n_iters):
        # Compute forces
        dots = points @ points.T
        np.fill_diagonal(dots, 1.0)
        dots = np.clip(dots, -1, 1)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.inf)
        
        # Force: repulsive, F = k * (1 - d) / d^2 * direction
        # Simpler: use gradient of -1/d^p
        p = 4
        k = 0.1
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Gradient of -k/d^p with respect to points[i]
                # d(-k/d^p)/d(points[i]) = k*p*(points[i] - (points[i]·points[j])*points[j]) / (1-(points[i]·points[j])^2) * 1/d^p
                # But on the sphere, we project to tangent space
                cos_d = dots[i, j]
                sin_d = np.sqrt(max(0, 1 - cos_d**2))
                if sin_d < 1e-10:
                    continue
                # Tangent component
                tangent = points[i] - cos_d * points[j]
                tangent = tangent / (np.linalg.norm(tangent) + 1e-15)
                force_mag = k * p / (d ** (p+1))
                forces[i] += force_mag * tangent
        
        # Velocity Verlet
        velocities = velocities + dt * forces
        points = points + dt * velocities
        points = _normalize(points)
        
        # Damping
        velocities *= 0.95
        
        # Track best
        energy = -np.sum(1.0 / np.where(dists < 1e-10, 1e10, dists)**p)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    return _normalize(best_points)


def _lloyd_refine(points, n_iters=100):
    """Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    points = _normalize(points.copy())
    
    for it in range(n_iters):
        # Compute Voronoi cells using Delaunay triangulation on sphere
        # For each point, find the centroid of its Voronoi region
        new_points = np.zeros_like(points)
        
        # Use a sampling-based approach for Voronoi centroids
        n_samples = 5000
        theta = np.random.uniform(0, np.pi, n_samples)
        phi = np.random.uniform(0, 2 * np.pi, n_samples)
        samples = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Assign each sample to nearest point
        dots = samples @ points.T  # (n_samples, N)
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroid for each point
        for i in range(N):
            mask = nearest == i
            if np.sum(mask) > 0:
                centroid = samples[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        
        # Check convergence
        if np.linalg.norm(new_points - points) < 1e-8:
            points = new_points
            break
        points = new_points
    
    return points


def _gradient_descent(points, n_iters=200, lr=0.01):
    """Gradient descent on the potential energy to minimize covering radius."""
    points = _normalize(points.copy())
    
    for it in range(n_iters):
        dots = points @ points.T
        np.fill_diagonal(dots, 1.0)
        dots = np.clip(dots, -1, 1)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.inf)
        
        p = 4
        k = 0.05
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                cos_d = dots[i, j]
                sin_d = np.sqrt(max(0, 1 - cos_d**2))
                if sin_d < 1e-10:
                    continue
                tangent = points[i] - cos_d * points[j]
                tangent = tangent / (np.linalg.norm(tangent) + 1e-15)
                force_mag = k * p / (d ** (p+1))
                forces[i] += force_mag * tangent
        
        points = points + lr * forces
        points = _normalize(points)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s limit
    
    best_points = None
    best_radius = np.inf
    
    # Multiple random restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points using Fibonacci spiral with random perturbation
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add random perturbation for different restarts
        if restart > 0:
            noise = np.random.randn(N, 3) * 0.1
            points = points + noise
            points = _normalize(points)
        
        # Phase 1: Repulsion-based refinement
        if time.time() > deadline:
            break
        points = _repulsion_refine(points, n_iters=300, dt=0.01)
        
        # Phase 2: Lloyd's algorithm
        if time.time() > deadline:
            break
        points = _lloyd_refine(points, n_iters=50)
        
        # Phase 3: Gradient descent polishing
        if time.time() > deadline:
            break
        points = _gradient_descent(points, n_iters=100, lr=0.01)
        
        # Evaluate
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
