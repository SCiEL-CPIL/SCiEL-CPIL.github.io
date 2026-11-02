"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_initial(N, n_iter=5000):
    """Generate a good initial hexagonal packing and return centers."""
    # Start with a hexagonal lattice
    # The optimal spacing for hex packing in a unit circle with N circles
    # We want to find a good initial configuration
    
    # Use a hexagonal lattice approach
    best_points = None
    best_score = 0.0
    
    for trial in range(10):
        # Random rotation for diversity
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Generate hexagonal lattice points
        # Spacing s such that we get roughly N points in unit circle
        # Area of unit circle = pi, area per hex cell = s^2 * sqrt(3)/2
        # N * s^2 * sqrt(3)/2 ≈ pi => s ≈ sqrt(2*pi / (N * sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0))) * 0.99
        
        # Generate lattice points
        R = 2.0 / s
        n = int(np.ceil(R))
        points = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x = i * s + (j % 2) * s / 2.0
                y = j * s * np.sqrt(3.0) / 2.0
                # Apply rotation
                xr = x * np.cos(angle) - y * np.sin(angle)
                yr = x * np.sin(angle) + y * np.cos(angle)
                if np.hypot(xr, yr) < 1.0 - s / 2.0:
                    points.append([xr, yr])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    return best_points


def _compute_score(points):
    """Compute the packing score: min(min_pairwise_dist/2, min(1 - |c_i|))."""
    n = len(points)
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair, min_boundary)


def _soft_min(dists, beta):
    """Soft minimum using log-sum-exp."""
    # dists is (N, N) matrix with inf on diagonal
    # We want soft min of off-diagonal elements
    # Use -log-sum-exp(-beta * dists) / beta
    mask = np.eye(len(dists), dtype=bool)
    d = dists.copy()
    d[mask] = 0  # Set diagonal to 0 so they don't contribute
    
    # For numerical stability
    d_clipped = np.clip(d, 1e-10, None)
    log_sum = np.log(np.sum(np.exp(-beta * d_clipped), axis=1))
    soft_min_vals = -log_sum / beta
    return np.min(soft_min_vals)


def _gradient_step(points, lr, beta, max_iter=100):
    """Perform gradient ascent on the soft-min objective."""
    n = len(points)
    p = points.copy()
    
    for it in range(max_iter):
        # Compute pairwise distances
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2) + 1e-16)
        
        # Soft min of pairwise distances (off-diagonal)
        mask = np.eye(n, dtype=bool)
        d_off = dists.copy()
        d_off[mask] = np.inf
        
        # Weight each pair by softmax
        exp_neg = np.exp(-beta * d_off)
        exp_neg[mask] = 0
        weights = exp_neg / (np.sum(exp_neg, axis=1, keepdims=True) + 1e-16)
        
        # Gradient of soft min w.r.t. positions
        # d(soft_min)/d(p_i) = sum_j weights_ij * (p_i - p_j) / |p_i - p_j|
        grad = np.zeros_like(p)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d_ij = dists[i, j]
                    if d_ij > 1e-10:
                        grad[i] += weights[i, j] * (p[i] - p[j]) / d_ij
        
        # Boundary constraint gradient
        radii = np.linalg.norm(p, axis=1)
        for i in range(n):
            if radii[i] > 1e-10:
                # Push inward if too close to boundary
                boundary_margin = 1.0 - radii[i]
                if boundary_margin < 0.01:
                    grad[i] -= p[i] / radii[i] * 0.5
        
        # Gradient ascent
        p = p + lr * grad
        
        # Project back into feasible region (simple projection)
        for i in range(n):
            r = np.linalg.norm(p[i])
            if r > 0.999:
                p[i] = p[i] * 0.999 / r
    
    return p


def _local_optimization(points, time_budget=60):
    """Local optimization using gradient-based methods with time budget."""
    start_time = time.time()
    p = points.copy()
    n = len(p)
    
    # Multi-scale optimization
    for beta in [10, 50, 100, 200, 500, 1000, 2000]:
        if time.time() - start_time > time_budget * 0.8:
            break
        
        lr = 0.001 / (1 + beta * 0.01)
        max_iter = 500 if beta < 100 else 200
        
        for it in range(max_iter):
            if time.time() - start_time > time_budget * 0.8:
                break
            
            # Compute pairwise distances
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2) + 1e-16)
            
            # Soft min weights
            mask = np.eye(n, dtype=bool)
            d_off = dists.copy()
            d_off[mask] = np.inf
            
            exp_neg = np.exp(-beta * d_off)
            exp_neg[mask] = 0
            weights = exp_neg / (np.sum(exp_neg, axis=1, keepdims=True) + 1e-16)
            
            # Gradient
            grad = np.zeros_like(p)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d_ij = dists[i, j]
                        if d_ij > 1e-10:
                            grad[i] += weights[i, j] * (p[i] - p[j]) / d_ij
            
            # Boundary repulsion
            radii = np.linalg.norm(p, axis=1)
            for i in range(n):
                if radii[i] > 1e-10:
                    margin = 1.0 - radii[i]
                    if margin < 0.05:
                        strength = 1.0 / (margin + 0.01)
                        grad[i] -= p[i] / radii[i] * strength * 0.1
            
            # Update
            p = p + lr * grad
            
            # Project to feasible region
            for i in range(n):
                r = np.linalg.norm(p[i])
                if r > 0.9999:
                    p[i] = p[i] * 0.9999 / r
            
            # Check for overlap and fix
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < 0.001:
                        # Separate them
                        if d > 1e-10:
                            push = (p[i] - p[j]) / d * 0.001
                            p[i] += push
                            p[j] -= push
                        else:
                            p[i][0] += 0.001
                            p[j][0] -= 0.001
    
    return p


def construct():
    """Construct the packing of 300 equal circles in a unit circle."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Step 1: Generate initial hexagonal packing
    points = _hexagonal_initial(N)
    
    if points is None or len(points) != N:
        # Fallback: simple grid
        points = np.zeros((N, 2))
        s = 0.1
        idx = 0
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * s
                y = j * s
                if np.hypot(x, y) < 0.95 and idx < N:
                    points[idx] = [x, y]
                    idx += 1
                if idx >= N:
                    break
            if idx >= N:
                break
    
    # Step 2: Local optimization with time budget
    time_spent = time.time() - start_time
    remaining = deadline - time_spent
    if remaining > 5:
        points = _local_optimization(points, time_budget=min(remaining - 2, 120))
    
    # Step 3: Final polish - ensure validity
    # Project all points to be strictly inside
    for i in range(N):
        r = np.linalg.norm(points[i])
        if r > 0.99999:
            points[i] = points[i] * 0.99999 / r
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-6:
                points[j][0] += 1e-5
                points[j][1] += 1e-5
    
    return points
# EVOLVE-BLOCK-END
