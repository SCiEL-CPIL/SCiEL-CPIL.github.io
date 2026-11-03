import numpy as np
import time
import math
from scipy.optimize import minimize, basin_hopping

N = 150
DEADLINE = 115.0  # Safety margin for 120s limit

def _subdivided_icosahedron(n_subdivisions):
    """Generate a subdivided icosahedron mesh on the unit sphere."""
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0
    # 12 vertices of the icosahedron
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    # Normalize
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 20 faces of the icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    for _ in range(n_subdivisions):
        new_faces = []
        midpoints = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                v1 = verts[i]
                v2 = verts[j]
                mid = (v1 + v2) / 2.0
                mid /= np.linalg.norm(mid)
                midpoints[key] = len(verts)
                verts = np.vstack([verts, mid])
            return midpoints[key]
        
        for f in faces:
            a, b, c = f
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        faces = new_faces
        
    return verts

def _log_energy_and_grad(pts):
    """Compute log energy and gradient for points on the sphere."""
    pts = pts.reshape(-N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist)
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    energy = np.sum(np.log(1.0 / d))
    
    # Gradient
    # dE/dx_i = sum_{j != i} - (x_i - x_j) / |x_i - x_j|^2
    # Vectorized:
    # grad_i = - sum_{j != i} (x_i - x_j) / d_{ij}^2
    # Note: d_{ij}^2 = |x_i - x_j|^2
    # We can compute the sum of (x_i - x_j) / d_{ij}^2 for all j.
    
    # To avoid division by zero and issues with diagonal, we set diagonal to 1.
    # But we need to exclude j=i in the sum.
    # Let's compute the full matrix of forces and zero out the diagonal.
    
    # inv_dist_sq = 1 / dist**2
    # inv_dist_sq[dist == 0] = 0 # Diagonal
    # forces = diff * inv_dist_sq[:, :, None]
    # grad = -np.sum(forces, axis=1)
    
    # However, dist has 1.0 on diagonal, so 1/1 = 1.
    # diff on diagonal is 0. So 0 * 1 = 0.
    # So we can just sum all rows.
    
    inv_dist_sq = np.ones_like(dist)
    mask = dist > 1e-12
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    forces = diff * inv_dist_sq[:, :, None]
    grad = -np.sum(forces, axis=1)
    
    # Project gradient to tangent space
    # grad_tangent = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    
    return energy, grad_tangent

def construct():
    start_time = time.time()
    
    # 1. Initialize points using a subdivided icosahedron (Geodesic Dome)
    # Level 2 subdivision yields 182 vertices. We need 150.
    # We can take a subset or use it as a guide.
    # For a robust start, we can use the vertices of a level 1 dome (62 points) 
    # and then add more? No, let's just generate a dense one and pick 150.
    # Actually, the vertices of a level 2 dome are very well distributed.
    # We can just take the first 150. Or better, we can use the fact that 
    # the dome is symmetric.
    
    try:
        dome_verts = _subdivided_icosahedron(2)
        # dome_verts has 182 points.
        # We need exactly 150.
        # We can randomly sample 150 from the 182 to break symmetry slightly 
        # or just take the first 150. Taking the first 150 might be biased 
        # due to the generation order. 
        # Let's shuffle and take 150.
        np.random.seed(42) # Fixed seed for reproducibility in init
        idx = np.random.choice(len(dome_verts), N, replace=False)
        init_pts = dome_verts[idx]
    except:
        # Fallback to Fibonacci if dome generation fails
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + math.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    points = init_pts.copy()
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    # 2. Define objective and gradient for the optimizer
    def objective_and_grad(x):
        pts = x.reshape(-N, 3)
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()

    # 3. Initial Local Minimization (L-BFGS-B)
    # This finds the nearest local minimum to the geodesic dome start.
    x0 = points.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    best_points = result.x.reshape(-N, 3)
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    best_energy, _ = _log_energy_and_grad(best_points)
    
    # 4. Basin Hopping
    # Basin hopping is ideal for this: it perturbs the configuration and then 
    # re-minimizes. This helps escape local minima that pure gradient descent 
    # or simple simulated annealing might get stuck in.
    
    # We need a custom perturbation function that moves points on the sphere.
    def perturb(x):
        # x is a flat array of 3N values.
        pts = x.reshape(-N, 3)
        # Randomly perturb a subset of points
        num_perturb = 10
        indices = np.random.choice(N, num_perturb, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            p = new_pts[idx]
            # Generate a random tangent vector
            rand_vec = np.random.randn(3)
            # Project to tangent space
            rand_vec -= np.dot(rand_vec, p) * p
            norm = np.linalg.norm(rand_vec)
            if norm > 1e-8:
                rand_vec = rand_vec / norm
                # Step size
                step = 0.1 
                new_p = p + step * rand_vec
                new_pts[idx] = new_p / np.linalg.norm(new_p)
        
        return new_pts.flatten()

    # Run basin hopping
    # maxval is the time limit in seconds for the basin hopping function itself?
    # No, scipy's basin_hopping doesn't have a direct time limit, but we can 
    # control iterations.
    # We will run it in a loop to respect the deadline.
    
    current_best_energy = best_energy
    current_best_points = best_points.copy()
    
    # We'll use a simple loop instead of scipy's basin_hopping to have tighter 
    # control over the time and to potentially use a more aggressive perturbation 
    # strategy if needed.
    # Actually, scipy's basin_hopping is quite efficient. Let's try to use it 
    # but we need to be careful about the time.
    # Since we don't have a time limit in the options, we will run a fixed 
    # number of iterations and check time periodically.
    
    # Let's implement a manual basin hopping loop for better control.
    temperature = 0.5
    cooling_rate = 0.95
    
    while time.time() - start_time < DEADLINE - 5.0:
        # Perturb
        perturbed_pts = best_points.copy()
        num_perturb = 10
        indices = np.random.choice(N, num_perturb, replace=False)
        for idx in indices:
            p = perturbed_pts[idx]
            rand_vec = np.random.randn(3)
            rand_vec -= np.dot(rand_vec, p) * p
            norm = np.linalg.norm(rand_vec)
            if norm > 1e-8:
                rand_vec = rand_vec / norm
                step = temperature
                new_p = p + step * rand_vec
                perturbed_pts[idx] = new_p / np.linalg.norm(new_p)
        
        # Local Minimize
        x0 = perturbed_pts.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
        
        new_pts = result.x.reshape(-N, 3)
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_energy, _ = _log_energy_and_grad(new_pts)
        
        # Metropolis Criterion
        delta_e = new_energy - current_best_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(temperature, 1e-6)):
            current_best_points = new_pts
            current_best_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_pts.copy()
        
        temperature *= cooling_rate
        
        # Check time
        if time.time() - start_time > DEADLINE - 5.0:
            break

    # 5. Final Polish
    if time.time() - start_time < DEADLINE - 2.0:
        x0 = best_points.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10})
        
        final_points = result.x.reshape(-N, 3)
        final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
        
        final_energy, _ = _log_energy_and_grad(final_points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points

    return best_points
