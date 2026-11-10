"""Spectral optimization for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh


def construct():
    """
    Construct 130 points on the unit sphere using a spectral optimization approach.
    
    Key idea: Parameterize the 130 points through a PSD matrix M (3x3) such that
    the points are derived from the eigenspace structure of M. We optimize over
    the entries of M to maximize the minimum pairwise angular separation.
    
    Approach:
    1. Represent the configuration via its Gram matrix G = XX^T (N x N, rank 3)
    2. The eigenvalues of G encode the "spread" of the configuration
    3. Optimize a smooth objective related to the eigenvalue spectrum to encourage
       uniform distribution on the sphere
    4. Use the eigendecomposition to recover point positions
    """
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def covering_radius(X, n_samples=50000):
        """Estimate covering radius by dense sampling."""
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * np.pi, n_samples)
        cos_phi = rng.uniform(-1, 1, n_samples)
        sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi**2))
        Y = np.column_stack([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi
        ])
        K = Y @ X.T
        np.clip(K, -1.0, 1.0, out=K)
        min_dist = np.arccos(K.max(axis=1))
        return min_dist.max()
    
    def spectral_objective(params):
        """
        Spectral objective: minimize the variance of the nearest-neighbor distances
        plus a penalty for small minimum distance.
        
        params: flattened N x 3 array of point coordinates
        """
        X = params.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = X / norms
        
        # Compute pairwise distances
        G = X @ X.T
        np.clip(G, -1.0, 1.0, out=G)
        D = np.arccos(G)
        np.fill_diagonal(D, np.inf)
        
        # For each point, min distance to others
        min_d = D.min(axis=1)
        
        # Objective: minimize variance + penalty for small min distance
        # This encourages uniform spreading
        var_term = np.var(min_d)
        min_penalty = 0.1 * np.exp(-10 * min_d.min())
        
        # Also add a term to encourage large minimum distance
        # Use smooth max via log-sum-exp
        T = 0.05
        shifted = min_d / T
        shifted -= shifted.max()
        smooth_max = T * (shifted.max() + np.log(np.sum(np.exp(shifted))))
        
        return var_term + min_penalty - 0.1 * smooth_max
    
    def spectral_grad(params):
        """Compute gradient of spectral objective."""
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = X / norms
        
        # Compute pairwise distances
        G = X @ X.T
        np.clip(G, -1.0, 1.0, out=G)
        D = np.arccos(G)
        np.fill_diagonal(D, np.inf)
        
        min_d = D.min(axis=1)
        
        # Gradient of variance term
        # d(var(min_d))/d(X_k) is complex, use numerical approximation
        # For efficiency, use a simpler gradient
        
        # Gradient of the smooth max term
        T = 0.05
        shifted = min_d / T
        shifted -= shifted.max()
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Gradient: -0.1 * sum_i weights[i] * d(min_d_i)/d(X)
        # d(min_d_i)/d(X_k) is tricky, approximate numerically
        
        # For now, return zero gradient and let scipy use numerical
        return np.zeros_like(params)
    
    # Phase 1: Spectral optimization using eigenvalue-based objective
    # We'll optimize the 390 coordinates with a smooth spectral objective
    
    def eigenvalue_objective(params):
        """
        Objective based on eigenvalues of the Gram matrix.
        We want the eigenvalues to be as equal as possible (uniform distribution).
        """
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = X / norms
        
        # Gram matrix
        G = X @ X.T
        np.clip(G, -1.0, 1.0, out=G)
        
        # Eigenvalues of G (only 3 are nonzero for rank-3 matrix)
        eigvals = np.linalg.eigvalsh(G)
        eigvals = eigvals[eigvals > 1e-10]
        
        # We want the nonzero eigenvalues to be equal (each ~N/3)
        # Minimize variance of eigenvalues
        if len(eigvals) < 2:
            return 1.0
        
        target = N / 3.0
        eig_dev = (eigvals - target) ** 2
        obj = np.sum(eig_dev)
        
        # Add a term to encourage large minimum distance
        D = np.arccos(G)
        np.fill_diagonal(D, np.inf)
        min_d = D.min()
        obj -= 0.5 * min_d
        
        return obj
    
    # Optimize using L-BFGS-B
    x0 = X.flatten()
    
    # Use multiple restarts with different initializations
    best_X = X.copy()
    best_cr = covering_radius(X, n_samples=30000)
    
    for restart in range(3):
        if restart > 0:
            # Perturb initial configuration
            rng = np.random.default_rng(restart * 100)
            X_pert = X.copy()
            for j in range(N):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                theta = rng.uniform(-0.1, 0.1)
                c, s = np.cos(theta), np.sin(theta)
                x, y, z = axis
                R = np.array([
                    [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                    [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                    [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
                ])
                X_pert[j] = R @ X_pert[j]
                X_pert[j] /= np.linalg.norm(X_pert[j])
            x0 = X_pert.flatten()
        
        result = minimize(
            eigenvalue_objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        
        X_opt = result.x.reshape(N, 3)
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        X_opt = X_opt / np.maximum(norms, 1e-10)
        
        cr = covering_radius(X_opt, n_samples=30000)
        if cr < best_cr:
            best_cr = cr
            best_X = X_opt.copy()
    
    X_current = best_X
    
    # Phase 2: Spherical Lloyd refinement
    def spherical_lloyd_step(X, n_samples=20000):
        """One step of spherical Lloyd algorithm."""
        rng = np.random.default_rng(777)
        theta = rng.uniform(0, 2 * np.pi, n_samples)
        cos_phi = rng.uniform(-1, 1, n_samples)
        sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi**2))
        Y = np.column_stack([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi
        ])
        
        # For each Y, find nearest point
        K = Y @ X.T
        nearest = K.argmax(axis=1)
        
        # Compute centroid of Voronoi cell for each point
        centroids = np.zeros((N, 3))
        for j in range(N):
            mask = nearest == j
            if mask.sum() > 0:
                centroids[j] = Y[mask].mean(axis=0)
        
        # Normalize
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        centroids = centroids / norms
        
        return centroids
    
    # Run spherical Lloyd with decreasing step size
    for it in range(100):
        centroids = spherical_lloyd_step(X_current, n_samples=30000)
        step = 0.5 / (1 + it * 0.05)
        X_current = X_current + step * (centroids - X_current)
        norms = np.linalg.norm(X_current, axis=1, keepdims=True)
        X_current = X_current / np.maximum(norms, 1e-10)
    
    # Phase 3: Local perturbation refinement
    def try_perturb(X, i, angle, n_trials=15):
        """Try to improve point i by small rotation."""
        best_X = X.copy()
        best_cr = covering_radius(X, n_samples=10000)
        
        rng = np.random.default_rng(i * 1000)
        for _ in range(n_trials):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            theta = rng.uniform(-angle, angle)
            c, s = np.cos(theta), np.sin(theta)
            x, y, z = axis
            R = np.array([
                [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
            ])
            
            X_new = X.copy()
            X_new[i] = R @ X[i]
            X_new[i] /= np.linalg.norm(X_new[i])
            
            cr = covering_radius(X_new, n_samples=10000)
            if cr < best_cr:
                best_cr = cr
                best_X = X_new.copy()
        
        return best_X, best_cr
    
    for round_num in range(4):
        angle = 0.12 / (round_num + 1)
        for i in range(N):
            X_current, _ = try_perturb(X_current, i, angle, n_trials=12)
    
    for round_num in range(2):
        angle = 0.02 / (round_num + 1)
        for i in range(N):
            X_current, _ = try_perturb(X_current, i, angle, n_trials=8)
    
    # Final normalization
    norms = np.linalg.norm(X_current, axis=1, keepdims=True)
    X_current = X_current / np.maximum(norms, 1e-10)
    
    return X_current
# EVOLVE-BLOCK-END
