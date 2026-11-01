"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping
import time

def _covering_radius(points):
    """Compute the covering radius in radians."""
    points = np.asarray(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Compute pairwise angular distances
    # To find the covering radius, we need the max over all points on the sphere of the min distance to a point in the set.
    # This is equivalent to the max circumradius of the convex hull facets.
    # However, computing the convex hull and circumradii is expensive and can be numerically unstable.
    # An alternative is to sample the sphere densely and check distances, but that's also slow.
    # The convex hull method is standard. Let's use scipy.spatial.ConvexHull.
    
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Fallback

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        # For a simplex on the unit sphere, the circumradius (angular distance from centroid/pole to vertex)
        # is the angle between the pole and a vertex.
        # Pole is the normalized normal to the facet.
        if len(verts) == 3:
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            max_rad = max(max_rad, theta)
        else:
            # Handle higher dimensional simplices if they occur (rare in 3D)
            # Use first 3 vertices
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            max_rad = max(max_rad, theta)
            
    return max_rad

def _energy(x):
    """Energy function for optimization."""
    pts = x.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    pts = pts / norms
    return _covering_radius(pts)

def _fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci spiral method."""
    phi = (1 + np.sqrt(5)) / 2 # Golden ratio
    golden_angle = 2 * np.pi / (phi ** 2)
    points = []
    for i in range(n):
        z = 1 - (i + 0.5) * 2 / n
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def construct():
    rng = np.random.default_rng(42)
    
    # Initial configuration: Fibonacci sphere
    x0 = _fibonacci_sphere(N).flatten()
    
    # Define local optimizer
    # Powell is a good derivative-free local optimizer
    # We use a tight tolerance for the local step
    local_optimizer = lambda x: __import__('scipy.optimize', fromlist=['minimize']).minimize(
        _energy, x, method='Powell', options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8}
    )
    
    # Run Basin Hopping
    # We limit the number of iterations to stay within time limits
    # Basin hopping is stochastic, so we set a seed for reproducibility if needed, 
    # but the rng in scipy.basinhopping uses its own.
    
    try:
        result = basinhopping(_energy, x0, niter=50, stepsize=0.5, 
                              minimizer_kwargs={'method': 'Powell'},
                              seed=42)
        x_opt = result.x
    except Exception:
        # Fallback to simple minimization if basinhopping fails
        from scipy.optimize import minimize
        result = minimize(_energy, x0, method='Powell', options={'maxiter': 5000})
        x_opt = result.x

    # Final refinement with a very tight Powell run
    from scipy.optimize import minimize
    final_result = minimize(_energy, x_opt, method='Powell', options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-10})
    
    points = final_result.x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
