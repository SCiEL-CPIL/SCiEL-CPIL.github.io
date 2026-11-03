"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms

def _covering_radius(X):
    """
    Compute the exact covering radius in degrees.
    The covering radius is the maximum over y on S^2 of min_i arccos(y . x_i).
    This maximum occurs at Voronoi vertices, which are equidistant to >= 3 points.
    For each triple (i,j,k), the Voronoi vertex is parallel to (x_i - x_j) x (x_i - x_k).
    We check both antipodal directions.
    """
    X = _project_to_sphere(X)
    n = len(X)
    max_dist = 0.0
    
    # Precompute all pair differences and cross products for speed
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = X[i] - X[j]
                v2 = X[i] - X[k]
                y = np.cross(v1, v2)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-10:
                    continue
                y = y / y_norm
                
                for sign in [1.0, -1.0]:
                    yc = sign * y
                    dots = X @ yc
                    max_dot = np.max(dots)
                    # Check that i, j, k are among the nearest points
                    if (abs(dots[i] - max_dot) < 1e-7 and 
                        abs(dots[j] - max_dot) < 1e-7 and 
                        abs(dots[k] - max_dot) < 1e-7):
                        dist = np.arccos(np.clip(max_dot, -1.0, 1.0))
                        if dist > max_dist:
                            max_dist = dist
    
    return np.degrees(max_dist)

def _soft_max_covering_radius(X, beta=100.0, M=3000):
    """
    Smooth approximation of the covering radius using soft-max over a grid.
    Returns (value, gradient) if requested, else just value.
    """
    X = _project_to_sphere(X)
    
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = Y @ X.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))  # (M,)
    
    max_d = np.max(distances)
    exp_terms = np.exp(beta * (distances - max_d))
    sum_exp = np.sum(exp_terms)
    soft_max = max_d + np.log(sum_exp) / beta
    
    return soft_max

def _soft_max_covering_radius_grad(X, beta=100.0, M=3000):
    """
    Smooth approximation of the covering radius with analytic gradient.
    Returns (value, gradient).
    """
    X = _project_to_sphere(X)
    n = len(X)
    
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = Y @ X.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    nearest_idx = np.argmax(dots, axis=1)  # (M,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))  # (M,)
    
    max_d = np.max(distances)
    exp_terms = np.exp(beta * (distances - max_d))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # (M,)
    
    soft_max = max_d + np.log(sum_exp) / beta
    
    # Gradient: d(soft_max)/dX_k = sum_m weights[m] * d(distances[m])/dX_k
    # d(arccos(d))/dX_k = -d / sqrt(1-d^2) where d = Y_m . X_k
    grad = np.zeros_like(X)
    for m in range(M):
        if weights[m] < 1e-15:
            continue
        d = max_dots[m]
        sqrt_term = np.sqrt(max(1.0 - d * d, 1e-15))
        coeff = -weights[m] * d / sqrt_term
        grad[nearest_idx[m]] += coeff * Y[m]
    
    return soft_max, grad

def _lloyd_iteration(X, iters=50):
    """
    Lloyd's algorithm for spherical k-means / covering.
    Each point moves to the centroid of its Voronoi cell (approximated by nearest-neighbor regions).
    """
    X = _project_to_sphere(X)
    n = len(X)
    
    for _ in range(iters):
        # Assign each point on a fine grid to its nearest X point
        M = 5000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Find nearest X point for each Y
        dots = Y @ X.T  # (M, N)
        nearest = np.argmax(dots, axis=1)  # (M,)
        
        # Compute centroid for each cluster
        new_X = np.zeros_like(X)
        for k in range(n):
            mask = (nearest == k)
            if np.sum(mask) > 0:
                centroid = np.mean(Y[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_X[k] = centroid / norm
                else:
                    new_X[k] = X[k]
            else:
                new_X[k] = X[k]
        
        X = _project_to_sphere(new_X)
    
    return X

def _simulated_annealing(X, start_temp=0.1, end_temp=0.001, n_iter=2000, deadline=None):
    """
    Simulated annealing to escape local minima.
    """
    X = _project_to_sphere(X)
    n = len(X)
    best_X = X.copy()
    best_score = _covering_radius(X)
    
    current_temp = start_temp
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and perturb it
        idx = np.random.randint(n)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_X = X.copy()
        new_X[idx] = new_X[idx] + current_temp * direction
        new_X = _project_to_sphere(new_X)
        
        # Evaluate using fast surrogate
        score = _soft_max_covering_radius(new_X, beta=50.0, M=1000)
        best_score_surr = _soft_max_covering_radius(best_X, beta=50.0, M=1000)
        
        if score < best_score_surr or np.random.rand() < np.exp(-(score - best_score_surr) / current_temp):
            X = new_X
        
        # Cool down
        current_temp = start_temp * (end_temp / start_temp) ** (it / n_iter)
        
        # Periodically evaluate exact score
        if it % 200 == 0:
            exact_score = _covering_radius(X)
            if exact_score < best_score:
                best_score = exact_score
                best_X = X.copy()
    
    # Final exact evaluation
    final_score = _covering_radius(X)
    if final_score < best_score:
        best_score = final_score
        best_X = X.copy()
    
    return best_X, best_score

def _lbfgs_refine(X, maxiter=300, beta=100.0, M=3000, deadline=None):
    """
    Refine using L-BFGS-B with the soft-max covering radius objective and analytic gradient.
    """
    X = _project_to_sphere(X)
    
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _project_to_sphere(X_local)
        val, grad = _soft_max_covering_radius_grad(X_local, beta=beta, M=M)
        return val, grad.flatten()
    
    try:
        result = minimize(
            objective_flat,
            X.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        if deadline is not None and time.time() > deadline:
            return X
        
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        return X_opt
    except Exception:
        return X

def _fibonacci_initialization(seed_offset=0.0):
    """Generate Fibonacci spiral points with optional jitter."""
    np.random.seed(int(seed_offset * 100000))
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.randn(N) * 0.05
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(X)

def _icosahedron_based_initialization():
    """Generate points based on icosahedron vertices with subdivision."""
    # 12 vertices of regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Start with icosahedron vertices and subdivide faces
    X = verts.copy()
    np.random.seed(42)
    
    # Add points by subdividing and projecting
    while len(X) < N:
        # Pick a random face and add its centroid
        idx = np.random.randint(0, 12)
        # Find 3 nearby points to form a triangle
        d = np.linalg.norm(X - X[idx], axis=1)
        d[idx] = np.inf
        j, k = np.argsort(d)[:2]
        centroid = (X[idx] + X[j] + X[k]) / 3.0
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm
            X = np.vstack([X, centroid])
    
    X = X[:N]
    # Add small random perturbation
    X += np.random.randn(N, 3) * 0.01
    return _project_to_sphere(X)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    
    best_X = None
    best_score = np.inf
    
    # Multi-start: try several diverse initializations
    initializations = [
        _fibonacci_initialization(0.0),
        _fibonacci_initialization(1.0),
        _icosahedron_based_initialization(),
    ]
    
    for init_idx, X_init in enumerate(initializations):
        if time.time() > deadline - 40:
            break
        
        X = X_init.copy()
        score = _covering_radius(X)
        
        # Lloyd relaxation
        X = _lloyd_iteration(X, iters=25)
        score = _covering_radius(X)
        
        # L-BFGS with moderate beta
        if time.time() < deadline - 30:
            X = _lbfgs_refine(X, maxiter=150, beta=50.0, M=2000, deadline=deadline - 25)
            score = _covering_radius(X)
        
        # Simulated annealing
        if time.time() < deadline - 25:
            X_sa, score_sa = _simulated_annealing(
                X, 
                start_temp=0.04, 
                end_temp=0.001, 
                n_iter=800,
                deadline=deadline - 20
            )
            if score_sa < score:
                score = score_sa
                X = X_sa
        
        # High-precision polish
        if time.time() < deadline - 15:
            X = _lbfgs_refine(X, maxiter=300, beta=500.0, M=4000, deadline=deadline - 10)
            score = _covering_radius(X)
        
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    # Final high-beta polish on best solution
    if best_X is not None and time.time() < deadline - 5:
        X = _lbfgs_refine(best_X, maxiter=200, beta=1000.0, M=5000, deadline=deadline - 2)
        score = _covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()
    
    if best_X is None:
        best_X = _fibonacci_initialization(0.0)
    
    return _project_to_sphere(best_X)

# EVOLVE-BLOCK-END
