import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 590  # Safety margin

    def score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def fibonacci_sphere(n):
        # Generate n points on a sphere using Fibonacci lattice
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def solve_sdp():
        """
        Solve SDP relaxation: maximize t subject to:
        - G is PSD (Gram matrix)
        - trace(G) = 1 (normalization)
        - G_ii = 1 for all i (unit norm points)
        - ||p_i - p_j||^2 >= t for all i < j
        
        This is equivalent to maximizing min pairwise distance squared.
        We then extract points via eigenvalue decomposition.
        """
        from scipy.optimize import linprog
        from scipy.linalg import eigh
        
        # We'll use a sequential approach:
        # 1. Start with Fibonacci sphere
        # 2. Use SDP-like optimization on Gram matrix
        
        # Initialize with Fibonacci sphere
        pts = fibonacci_sphere(n)
        
        # Compute initial Gram matrix
        G = pts @ pts.T
        
        # Optimize: we want to maximize the minimum off-diagonal element of (2*I - G)
        # which corresponds to squared distances
        
        # Use a custom optimization on the Gram matrix
        # Variables: G flattened (upper triangle only to ensure symmetry)
        # Constraints: G PSD, trace(G) = 1, G_ii = 1
        
        def gram_to_points(G_mat):
            """Extract points from Gram matrix via eigenvalue decomposition"""
            # Ensure symmetry
            G_mat = (G_mat + G_mat.T) / 2
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = eigh(G_mat)
            # Take top d eigenvalues/vectors
            idx = np.argsort(eigenvalues)[::-1][:d]
            eigenvalues_d = eigenvalues[idx]
            eigenvectors_d = eigenvectors[:, idx]
            # Points are rows of eigenvectors_d * sqrt(eigenvalues_d)
            mask = eigenvalues_d > 1e-10
            if np.sum(mask) < d:
                return None
            eigenvalues_d = eigenvalues_d[mask]
            eigenvectors_d = eigenvectors_d[:, mask]
            pts = eigenvectors_d * np.sqrt(np.maximum(eigenvalues_d, 0))
            return pts
        
        def sdp_objective(x):
            """Objective: minimize negative of soft-min of squared distances"""
            # Reconstruct Gram matrix from upper triangle
            G = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    G[i, j] = x[idx]
                    G[j, i] = x[idx]
                    idx += 1
            
            # Check PSD
            eigvals = np.linalg.eigvalsh(G)
            if np.min(eigvals) < -1e-6:
                return 1e6  # Penalty for non-PSD
            
            # Enforce unit diagonal
            diag_penalty = np.sum((np.diag(G) - 1.0) ** 2)
            if diag_penalty > 1e-4:
                return 1e6 + diag_penalty
            
            # Compute squared distances from Gram matrix
            # ||p_i - p_j||^2 = G_ii + G_jj - 2*G_ij = 2 - 2*G_ij (since G_ii = 1)
            dist_sq = 2.0 - 2.0 * G
            np.fill_diagonal(dist_sq, 0.0)
            
            # Soft-min using log-sum-exp
            beta = 50.0
            min_dist_sq = -np.log(np.sum(np.exp(-beta * dist_sq))) / beta
            
            return -min_dist_sq + 0.1 * diag_penalty
        
        # Initial point: upper triangle of Fibonacci Gram matrix
        G_init = pts @ pts.T
        x0 = np.zeros(n * (n + 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                x0[idx] = G_init[i, j]
                idx += 1
        
        # Optimize
        try:
            res = minimize(
                sdp_objective,
                x0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res.fun < 1e5:  # Valid solution
                # Extract points
                G_opt = np.zeros((n, n))
                idx = 0
                for i in range(n):
                    for j in range(i, n):
                        G_opt[i, j] = res.x[idx]
                        G_opt[j, i] = res.x[idx]
                        idx += 1
                
                # Ensure PSD
                eigvals, eigvecs = eigh(G_opt)
                eigvals = np.maximum(eigvals, 0)
                G_opt = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Extract points
                pts_opt = gram_to_points(G_opt)
                if pts_opt is not None and np.all(np.isfinite(pts_opt)):
                    return pts_opt
        except Exception:
            pass
        
        return pts

    best_points = None
    best_score = 0.0
    
    # Try SDP approach first
    sdp_pts = solve_sdp()
    if sdp_pts is not None:
        sdp_score = score(sdp_pts)
        best_points = sdp_pts
        best_score = sdp_score
    
    # Fallback: Fibonacci sphere
    init_pts = fibonacci_sphere(n)
    fib_score = score(init_pts)
    if fib_score > best_score:
        best_points = init_pts
        best_score = fib_score
    
    # Polish with local optimization
    def objective(x):
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists = pdist(p)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1.0
        d_min = np.min(dists)
        return -(d_min / d_max) ** 2

    rng = np.random.default_rng(42)
    
    # Multi-start local optimization for polishing
    restarts = 0
    max_restarts = 50
    
    while time.time() - start_time < time_limit - 10 and restarts < max_restarts:
        if restarts == 0:
            x0 = best_points.ravel()
        else:
            # Perturb best points
            x0 = best_points.ravel() + rng.normal(0, 0.1, size=(n, d)).ravel()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            restarts += 1
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
                
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
        
    return best_points
