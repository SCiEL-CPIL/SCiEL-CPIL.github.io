"""Physics-based relaxation for packing 300 equal circles in a unit circle."""
import numpy as np

N = 300


def _hex_lattice_init(n_points, radius=1.0):
    """Generate points on a hexagonal lattice inside a circle of given radius."""
    points = []
    # Use a hexagonal lattice with spacing chosen to fit approximately n_points
    # Start with an estimate and refine
    # For hex packing in a circle of radius R, number of points ~ pi * R^2 / (sqrt(3)/2 * s^2)
    # So s ~ sqrt(2 * pi * R^2 / (sqrt(3) * n_points))
    s = np.sqrt(2.0 * np.pi * radius**2 / (np.sqrt(3.0) * n_points))
    
    # Generate lattice points
    max_k = int(np.ceil(radius / s)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x * x + y * y <= radius * radius:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        # Fallback: uniform random
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        r = np.sqrt(np.random.uniform(0, 1, n_points))
        points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    elif len(points) < n_points:
        # Need more points, decrease spacing
        for _ in range(20):
            s *= 0.95
            points = []
            max_k = int(np.ceil(radius / s)) + 2
            for j in range(-max_k, max_k + 1):
                for i in range(-max_k, max_k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if x * x + y * y <= radius * radius:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= n_points:
                break
        if len(points) < n_points:
            # Fill with random points
            while len(points) < n_points:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 1))
                points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    # Select n_points points, preferring those closer to center for stability
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n_points]
    return points[idx]


def construct():
    """
    Construct centers of N=300 equal circles inside a unit circle.
    Uses physics-based relaxation with soft penalties for overlap and boundary violation.
    Returns array of shape (N, 2).
    """
    np.random.seed(42)
    
    # Initial configuration from hexagonal lattice
    points = _hex_lattice_init(N, radius=0.98)
    
    # Simulation parameters
    n_iters = 5000
    dt = 0.001
    temperature = 0.01
    cooling_rate = 0.9995
    repulsion_strength = 10.0
    boundary_strength = 50.0
    max_force = 1.0
    
    best_points = points.copy()
    best_score = -np.inf
    
    for it in range(n_iters):
        # Compute forces
        n = len(points)
        forces = np.zeros_like(points)
        
        # Pairwise repulsion (soft overlap penalty)
        # Use a soft repulsion: F ~ 1/d^2 when d < some threshold
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    # Coincident points, push apart randomly
                    diff = np.random.normal(size=2)
                    d = np.linalg.norm(diff)
                
                # Target distance is 2*r, but we don't know r yet.
                # Instead, use a repulsion that pushes points apart when too close.
                # We want to maximize the minimum distance, so repel when d is small.
                # Use a potential V = 1/d^2, so F = 2/d^3 * diff
                # But this is too strong. Use a softer potential.
                # V = exp(-d^2 / sigma^2) or similar
                # Let's use V = 1/(d^2 + eps) for numerical stability
                eps = 1e-6
                inv_d2 = 1.0 / (d * d + eps)
                f_mag = repulsion_strength * inv_d2
                f_vec = f_mag * diff / d
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Boundary force: push points inward if they're outside the unit circle
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 0.99:  # Close to boundary
                # Soft boundary: force pushing inward proportional to how far outside
                excess = r - 0.99
                if excess > 0:
                    f_mag = boundary_strength * excess
                    f_vec = -f_mag * points[i] / r
                    forces[i] += f_vec
                else:
                    # Even inside, apply a gentle inward pull to keep them well inside
                    f_mag = boundary_strength * 0.1 * (1.0 - r)
                    f_vec = -f_mag * points[i] / (r + 1e-10)
                    forces[i] += f_vec
        
        # Cap forces
        for i in range(n):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > max_force:
                forces[i] *= max_force / f_norm
        
        # Langevin update
        noise = np.random.normal(size=points.shape) * temperature
        points += dt * forces + noise
        
        # Keep points inside the container (hard constraint)
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 0.999:
                points[i] *= 0.999 / r
        
        # Compute score
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        min_boundary = np.inf
        for i in range(n):
            r = np.linalg.norm(points[i])
            boundary_dist = 1.0 - r
            if boundary_dist < min_boundary:
                min_boundary = boundary_dist
        
        score = min(min_dist / 2.0, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 1e-6:
            temperature = 1e-6
    
    return best_points
