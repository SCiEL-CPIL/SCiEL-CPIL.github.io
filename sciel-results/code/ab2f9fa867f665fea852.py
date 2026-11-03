"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

# Unit equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    """Check if point p is inside or on the boundary of triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    """Objective function for minimization: negative of minimum triangle area."""
    points = x.reshape(-1, 2)
    
    # Check if all points are inside the triangle
    for p in points:
        if not point_in_triangle(p):
            return -1e-10  # Very small value to discourage going outside
    
    return -min_triangle_area(points)

def project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    if point_in_triangle(p):
        return p
    
    # Find closest point on the triangle boundary
    best = p
    best_dist = float('inf')
    
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        # Project p onto line segment ab
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    
    # Move slightly inward
    centroid = (V0 + V1 + V2) / 3
    best = 0.99 * best + 0.01 * centroid
    
    return best

def generate_grid_initialization(n_grid=6):
    """Generate diverse starting points using a grid with structured jitter."""
    starts = []
    
    # Create a grid of points inside the triangle
    for i in range(n_grid):
        for j in range(n_grid - i):
            # Barycentric coordinates
            u = (i + 0.5) / n_grid
            v = (j + 0.5) / (n_grid - i)
            
            # Convert to Cartesian
            x = u * (1 - v) + 0.5 * v
            y = (np.sqrt(3) / 2) * v
            
            # Add structured jitter based on position
            jitter_scale = 0.1
            dx = jitter_scale * np.sin(i * 1.7 + j * 2.3)
            dy = jitter_scale * np.cos(i * 2.1 + j * 1.9)
            
            p = np.array([x + dx, y + dy])
            p = project_to_triangle(p)
            
            # Create a configuration by spreading points around this center
            config = np.zeros((N, 2))
            for k in range(N):
                angle = 2 * np.pi * k / N
                r = 0.15 + 0.05 * (k % 3)
                config[k] = p + r * np.array([np.cos(angle), np.sin(angle)])
                config[k] = project_to_triangle(config[k])
            
            starts.append(config.flatten())
    
    return starts

def nelder_mead_optimize(x0, maxiter=500, tol=1e-6):
    """Run Nelder-Mead optimization."""
    try:
        result = minimize(
            negative_min_area_flat,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x
    except Exception:
        return x0

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the minimum triangle area."""
    
    # Generate diverse starting points
    starts = generate_grid_initialization(n_grid=5)
    
    # Also add a few random starts for diversity
    np.random.seed(42)
    for _ in range(5):
        config = np.zeros((N, 2))
        for k in range(N):
            # Random point in triangle using barycentric coordinates
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            config[k] = np.array([u + 0.5 * v, (np.sqrt(3) / 2) * v])
        starts.append(config.flatten())
    
    # Run multi-start Nelder-Mead optimization
    best_x = None
    best_score = -float('inf')
    
    for x0 in starts:
        # Ensure starting points are valid
        x0_points = x0.reshape(-1, 2)
        for i in range(N):
            x0_points[i] = project_to_triangle(x0_points[i])
        x0 = x0_points.flatten()
        
        # Run optimization
        x_opt = nelder_mead_optimize(x0, maxiter=300)
        
        # Evaluate
        points = x_opt.reshape(-1, 2)
        # Project any points that might have gone outside
        for i in range(N):
            points[i] = project_to_triangle(points[i])
        
        score = min_triangle_area(points)
        
        if score > best_score:
            best_score = score
            best_x = points.copy()
    
    return best_x
# EVOLVE-BLOCK-END
