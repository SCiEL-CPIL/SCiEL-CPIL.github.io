"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_distances(c):
    """Compute all pairwise distances for N points."""
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    # Use broadcasting for pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist

def _score(c):
    """Compute the radius score for a configuration."""
    c = np.asarray(c, dtype=np.float64)
    # Boundary distances
    boundary = 0.5 - np.abs(c)  # shape (N, 3)
    min_boundary = np.min(boundary, axis=1)  # shape (N,)
    
    # Pairwise distances
    dist = _pairwise_distances(c)
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist, axis=1)  # shape (N,)
    
    # The radius is half of the minimum of all constraints
    all_constraints = np.concatenate([min_boundary, min_pair / 2.0])
    r = np.min(all_constraints)
    return r

def _negative_score(c):
    """Negative score for minimization."""
    return -_score(c)

def _objective_smooth(c, beta=100.0):
    """
    Smooth surrogate using log-sum-exp for the min.
    We want to maximize min(constraints), so minimize -log-sum-exp(-constraints/beta) * beta
    Actually, let's use a soft-min approach.
    min_i x_i ≈ -1/beta * log(sum_i exp(-beta * x_i))
    We want to maximize min(x_i), so we minimize -min(x_i) ≈ 1/beta * log(sum_i exp(-beta * x_i))
    """
    c = np.asarray(c, dtype=np.float64)
    boundary = 0.5 - np.abs(c)
    min_boundary = np.min(boundary, axis=1)
    
    dist = _pairwise_distances(c)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist, axis=1) / 2.0
    
    all_constraints = np.concatenate([min_boundary, min_pair])
    
    # Soft-min
    beta = 100.0
    # To avoid overflow, shift by max
    m = np.max(all_constraints)
    soft_min = -m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_constraints - m))))
    
    return -soft_min  # We minimize this, so we maximize soft_min

def _init_fcc(n_points):
    """
    Initialize points on an FCC-like lattice.
    FCC lattice points in a cube.
    """
    points = []
    # FCC basis vectors
    a = 0.5  # lattice constant for unit cube
    
    # Generate FCC points
    # FCC unit cell has points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # in units of the lattice constant
    
    # Let's use a simpler approach: start with a grid and perturb
    m = int(np.ceil(n_points ** (1.0 / 3.0)))
    
    # Create a grid of candidate positions
    x = np.linspace(-0.5 + 0.05, 0.5 - 0.05, m)
    y = np.linspace(-0.5 + 0.05, 0.5 - 0.05, m)
    z = np.linspace(-0.5 + 0.05, 0.5 - 0.05, m)
    
    # Generate all grid points
    candidates = np.array([(xi, yi, zi) for xi in x for yi in y for zi in z])
    
    # Add some FCC-like points
    fcc_points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                xi = -0.5 + (i + 0.5) * (1.0 / m)
                yi = -0.5 + (j + 0.5) * (1.0 / m)
                zi = -0.5 + (k + 0.5) * (1.0 / m)
                fcc_points.append((xi, yi, zi))
                # FCC offsets
                if i < m - 1 and j < m - 1:
                    fcc_points.append((xi + 0.5/m, yi + 0.5/m, zi))
                if i < m - 1 and k < m - 1:
                    fcc_points.append((xi + 0.5/m, yi, zi + 0.5/m))
                if j < m - 1 and k < m - 1:
                    fcc_points.append((xi, yi + 0.5/m, zi + 0.5/m))
    
    all_candidates = np.array(fcc_points + list(candidates))
    
    # Randomly select n_points from candidates
    idx = np.random.choice(len(all_candidates), n_points, replace=False)
    return all_candidates[idx].copy()

def _init_grid(n_points):
    """Initialize with a simple grid."""
    m = int(np.ceil(n_points ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_points].copy()

def _refine(c, max_iter=500, tol=1e-12):
    """
    Refine a configuration using gradient-based optimization.
    """
    c = np.asarray(c, dtype=np.float64).flatten().copy()
    n = len(c) // 3
    
    best_c = c.copy()
    best_score = _score(c.reshape(-1, 3))
    
    for _ in range(max_iter):
        # Compute gradient numerically
        eps = 1e-7
        grad = np.zeros_like(c)
        for i in range(len(c)):
            c_plus = c.copy()
            c_plus[i] += eps
            f_plus = _negative_score(c_plus.reshape(-1, 3))
            
            c_minus = c.copy()
            c_minus[i] -= eps
            f_minus = _negative_score(c_minus.reshape(-1, 3))
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient descent with adaptive step size
        step = 0.01
        for _ in range(10):
            new_c = c - step * grad
            # Project back to cube
            new_c = np.clip(new_c, -0.5, 0.5)
            
            new_score = _score(new_c.reshape(-1, 3))
            if new_score > best_score:
                best_score = new_score
                best_c = new_c.copy()
                c = new_c.copy()
                break
            else:
                step *= 0.5
        
        if step < 1e-10:
            break
    
    return best_c.reshape(-1, 3), best_score

def construct():
    """
    Construct 50 sphere centers in a unit cube to maximize the minimum radius.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    best_c = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [
        ("grid", _init_grid(N)),
        ("fcc", _init_fcc(N)),
    ]
    
    # Add random initializations
    for i in range(5):
        random_c = np.random.uniform(-0.4, 0.4, (N, 3))
        inits.append((f"random_{i}", random_c))
    
    for name, init_c in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Refine
        refined_c, score = _refine(init_c, max_iter=200, tol=1e-10)
        
        if score > best_score:
            best_score = score
            best_c = refined_c.copy()
    
    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 5.0:
        polished_c, polished_score = _refine(best_c, max_iter=500, tol=1e-12)
        if polished_score > best_score:
            best_score = polished_score
            best_c = polished_c.copy()
    
    # Ensure valid output
    if best_c is None:
        best_c = _init_grid(N)
    
    # Clip to ensure points are within the cube
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
