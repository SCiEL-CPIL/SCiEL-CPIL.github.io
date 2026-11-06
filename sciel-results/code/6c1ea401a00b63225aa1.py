"""Physics-based relaxation for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 50 equal circles in a unit circle using physics-based relaxation.
    """
    np.random.seed(42)
    
    def energy_and_grad(x):
        """
        Compute the energy and gradient for the physics-based potential.
        We use a strong repulsive potential to maximize minimum distance.
        """
        x = x.reshape(N, 2)
        
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Avoid division by zero on diagonal
        dist_diag = np.where(np.eye(N, dtype=bool), 1e10, dist)
        
        # Repulsive potential: sum over i<j of 1/d^6
        # Using d^-6 makes the repulsion very stiff at short ranges
        iu = np.triu_indices(N, k=1)
        pair_dist_sq = dist_sq[iu]
        pair_dist_sq = np.maximum(pair_dist_sq, 1e-12)
        # 1/d^6 = 1/(d^2)^3
        repulsion = np.sum(1.0 / pair_dist_sq**3)
        
        # Wall potential: sum over i of 1/(1-|c_i|)^6
        r = np.sqrt(np.sum(x**2, axis=1))
        wall_margin = np.maximum(1.0 - r, 1e-8)
        # 1/(1-r)^6 = 1/((1-r)^2)^3
        wall = np.sum(1.0 / wall_margin**6)
        
        # Total energy
        E = repulsion + wall
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Gradient of repulsion: d(1/d^6)/dx = -6/d^7 * (x_i - x_j)/d = -6(x_i-x_j)/d^8
        # In terms of d^2: d((d^2)^-3)/dx = -3(d^2)^-4 * 2(x_i-x_j) = -6(x_i-x_j)/(d^2)^4
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d2 = pair_dist_sq[idx]
            if d2 < 1e-12:
                continue
            coeff = -6.0 / (d2**4)
            diff_ij = x[i] - x[j]
            grad[i] += coeff * diff_ij
            grad[j] -= coeff * diff_ij
        
        # Gradient of wall potential: d(1/(1-r)^6)/dx = 6/(1-r)^7 * x/r
        for i in range(N):
            ri = r[i]
            if ri < 1e-8:
                continue
            margin = 1.0 - ri
            if margin < 1e-8:
                margin = 1e-8
            coeff = 6.0 / (margin**7) * 1.0 / ri
            grad[i] += coeff * x[i]
        
        return E, grad.ravel()
    
    def score(x):
        """Compute the actual score: the largest common radius."""
        x = x.reshape(N, 2)
        r = np.sqrt(np.sum(x**2, axis=1))
        if np.any(r > 1.0):
            return 0.0
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e10)
        min_pair_dist = np.min(dist)
        
        min_wall_dist = np.min(1.0 - r)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def try_initialization(init_x):
        """Run optimization from a given initial position."""
        init_x = init_x.reshape(-1)
        
        result = minimize(
            energy_and_grad,
            init_x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 10000}
        )
        
        return result.x
    
    def generate_hex_init(scale, rotation):
        """Generate a hexagonal lattice initialization."""
        points = []
        # Hex grid spacing
        a = scale
        # Iterate over a grid large enough to cover the unit circle
        for j in range(-10, 11):
            for i in range(-10, 11):
                px = a * (i + 0.5 * (j % 2))
                py = a * j * np.sqrt(3.0) / 2.0
                # Rotate
                px_rot = px * np.cos(rotation) - py * np.sin(rotation)
                py_rot = px * np.sin(rotation) + py * np.cos(rotation)
                if px_rot**2 + py_rot**2 <= 1.0:
                    points.append([px_rot, py_rot])
        
        points = np.array(points)
        if len(points) < N:
            # Add random points if not enough
            while len(points) < N:
                r = np.sqrt(np.random.uniform(0, 1)) * 0.9
                theta = np.random.uniform(0, 2 * np.pi)
                points = np.vstack([points, [r * np.cos(theta), r * np.sin(theta)]])
        else:
            # Keep N closest to center? Or random N?
            # Closest to center is usually better for dense packing
            r = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(r)[:N]
            points = points[idx]
            
        return points

    best_score = -1
    best_x = None
    
    # Strategy 1: Hexagonal Lattice with various scales and rotations
    # The optimal radius is approx 0.1258. Diameter ~ 0.2516.
    # Hex grid spacing should be around 0.25.
    scales = [0.24, 0.25, 0.26, 0.27]
    rotations = [0, np.pi/6, np.pi/3, np.pi/2]
    
    for scale in scales:
        for rot in rotations:
            init_x = generate_hex_init(scale, rot)
            x_opt = try_initialization(init_x)
            s = score(x_opt)
            if s > best_score:
                best_score = s
                best_x = x_opt
    
    # Strategy 2: Random initializations
    for attempt in range(10):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
        init_x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        x_opt = try_initialization(init_x)
        s = score(x_opt)
        if s > best_score:
            best_score = s
            best_x = x_opt
    
    # Strategy 3: Local refinement / Jiggle
    if best_x is not None:
        for _ in range(5):
            # Perturb
            noise = np.random.normal(0, 0.01, best_x.shape)
            x_perturbed = best_x + noise
            # Ensure inside
            r = np.sqrt(np.sum(x_perturbed**2, axis=1))
            mask = r > 0.99
            x_perturbed[mask] *= 0.99 / r[mask]
            
            x_opt = try_initialization(x_perturbed)
            s = score(x_opt)
            if s > best_score:
                best_score = s
                best_x = x_opt
            else:
                # If it didn't improve, try a smaller perturbation
                noise = np.random.normal(0, 0.001, best_x.shape)
                x_perturbed = best_x + noise
                r = np.sqrt(np.sum(x_perturbed**2, axis=1))
                mask = r > 0.99
                x_perturbed[mask] *= 0.99 / r[mask]
                x_opt = try_initialization(x_perturbed)
                s = score(x_opt)
                if s > best_score:
                    best_score = s
                    best_x = x_opt

    # Final polish
    if best_x is not None:
        result = minimize(
            energy_and_grad,
            best_x.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 20000}
        )
        x_final = result.x.reshape(N, 2)
        s_final = score(x_final)
        if s_final > best_score:
            best_x = x_final
            best_score = s_final
    
    # Ensure all points are strictly inside
    r = np.sqrt(np.sum(best_x**2, axis=1))
    for i in range(N):
        if r[i] >= 1.0:
            best_x[i] *= 0.999
    
    return best_x

# EVOLVE-BLOCK-END
