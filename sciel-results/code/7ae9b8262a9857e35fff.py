"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, Bounds
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_score(centers):
    """Compute the score: min of half pairwise distances and min distance to boundary."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    # Distance to boundary: min over each coordinate of (0.5 - |c|)
    # For cube [-0.5, 0.5]^3
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    return min(min_pairwise, min_boundary)

def _negative_score(centers):
    return -_compute_score(centers)

def _initial_bcc():
    """Generate BCC-like initial configuration for 50 spheres."""
    # Use a grid-based approach with some randomization
    # 3x4x5 = 60, take 50
    xs = np.linspace(-0.4, 0.4, 3)
    ys = np.linspace(-0.4, 0.4, 4)
    zs = np.linspace(-0.4, 0.4, 5)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    
    points = np.array(points[:50])
    # Add small random perturbation
    np.random.seed(42)
    points += np.random.randn(50, 3) * 0.01
    return points

def _initial_grid():
    """Generate grid-based initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _anneal(centers, n_steps=500, temp_start=0.05, temp_end=0.001):
    """Repulsion annealing to push spheres apart."""
    centers = centers.copy()
    n = len(centers)
    
    for step in range(n_steps):
        # Annealing schedule
        t = temp_start * (temp_end / temp_start) ** (step / max(n_steps - 1, 1))
        
        # Compute forces
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3) * 1e-6
                    dist = np.linalg.norm(d)
                # Repulsive force ~ 1/dist^2
                f = d / (dist ** 2) * t
                forces[i] += f
                forces[j] -= f
        
        # Boundary repulsion
        for i in range(n):
            for k in range(3):
                if centers[i, k] > 0:
                    forces[i, k] -= (0.5 - centers[i, k]) ** 2 * t * 10
                else:
                    forces[i, k] += (0.5 + centers[i, k]) ** 2 * t * 10
        
        # Update positions with random noise
        centers += forces
        centers += np.random.randn(n, 3) * t * 0.1
        
        # Keep inside cube
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations
    best_centers = None
    best_score = -np.inf
    
    initializers = [
        _initial_grid,
        _initial_bcc,
    ]
    
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        centers = init_func()
        
        # Annealing phase
        if time.time() < deadline - 30:
            centers = _anneal(centers, n_steps=200, temp_start=0.03, temp_end=0.0005)
        
        # Gradient-based optimization
        if time.time() < deadline - 10:
            bounds = Bounds(-0.499, 0.499)
            result = minimize(
                _negative_score,
                centers,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success or result.fun < best_score * -1:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_centers = result.x.copy()
        
        if best_centers is None:
            best_centers = centers
            best_score = _compute_score(centers)
    
    # Final polish: high-precision local optimization
    if best_centers is not None and time.time() < deadline - 5:
        bounds = Bounds(-0.4999, 0.4999)
        result = minimize(
            _negative_score,
            best_centers,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        score = -result.fun
        if score > best_score:
            best_score = score
            best_centers = result.x.copy()
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _initial_grid()
    
    # Clip to ensure inside cube
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    # Final validation
    final_score = _compute_score(best_centers)
    if final_score <= 0:
        # Fallback to grid
        best_centers = _initial_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
