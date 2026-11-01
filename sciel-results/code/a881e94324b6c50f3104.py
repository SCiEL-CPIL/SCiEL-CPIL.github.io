import numpy as np
import time
import math

N = 150

def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init():
    # Simple Halton sequence in 2D, mapped to sphere
    # Using a simple prime base sequence
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, N + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        # Map [0,1]x[0,1] to sphere
        # Use spherical coords
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])
    return np.array(points)

def _energy_and_grad(X):
    # X is (N, 3)
    # Compute pairwise distances
    # D^2_{ij} = |Xi|^2 + |Xj|^2 - 2 Xi.Xj = 2 - 2 Xi.Xj
    # Since |Xi|=1
    G = X @ X.T
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, 1.0) # Avoid log(0)
    
    # Energy: -sum log(D) for i<j
    # D = sqrt(D2)
    # log(D) = 0.5 log(D2)
    # E = -0.5 * sum_{i<j} log(D2_{ij})
    # Sum over i<j is 0.5 * sum_{i!=j}
    # So E = -0.5 * 0.5 * sum_{i!=j} log(D2_{ij}) = -0.25 * sum_{i!=j} log(D2_{ij})
    # Wait.
    # E = sum_{i<j} log(1/D_{ij}) = - sum_{i<j} log D_{ij}
    # log D_{ij} = 0.5 log D2_{ij}
    # sum_{i<j} 0.5 log D2_{ij} = 0.5 * (0.5 * sum_{i!=j} log D2_{ij}) = 0.25 sum_{i!=j} log D2_{ij}
    # So E = -0.25 * sum_{i!=j} log(D2_{ij})
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    log_D2 = np.log(D2)
    log_D2[~mask] = 0.0
    energy = -0.25 * np.sum(log_D2)
    
    # Gradient
    # grad_i E = sum_{j!=i} (x_i - x_j) / D2_{ij}
    # Let's compute this matrix.
    # G_ij = (x_i - x_j) / D2_{ij}
    # We can compute this as:
    # num = X[:, None, :] - X[None, :, :]
    # grad_mat = num / D2[:, :, None]
    # Then grad_i = sum_j grad_mat[i,j]
    
    # Optimization:
    # grad_i = sum_{j!=i} x_i / D2_{ij} - sum_{j!=i} x_j / D2_{ij}
    # = x_i * sum_{j!=i} 1/D2_{ij} - sum_{j!=i} x_j / D2_{ij}
    
    inv_D2 = 1.0 / D2
    np.fill_diagonal(inv_D2, 0.0)
    
    sum_inv = np.sum(inv_D2, axis=1) # (N,)
    sum_xj_over_D2 = inv_D2 @ X # (N, 3)
    
    euclid_grad = X * sum_inv[:, None] - sum_xj_over_D2
    
    # Project to tangent space
    # g_i = euclid_grad_i - (euclid_grad_i . x_i) x_i
    dot = np.sum(euclid_grad * X, axis=1) # (N,)
    rgrad = euclid_grad - dot[:, None] * X
    
    return energy, rgrad

def _exp_map(X, V, t):
    # X: (N, 3), V: (N, 3) tangent vectors
    # t: scalar step size
    # exp_X(tV) = cos(t|V|) X + sin(t|V|)/|V| (tV)
    # Note: V is already scaled by t? No, let's keep t separate for line search.
    # Actually, usually we pass direction D and step t.
    # Here V is the direction. We want to move by t*V.
    # Let W = t * V
    # exp_X(W) = cos(|W|) X + sin(|W|)/|W| W
    
    W = t * V
    norms = np.linalg.norm(W, axis=1)
    # Handle small norms
    cos_norms = np.cos(norms)
    sin_norms = np.sin(norms)
    # Avoid division by zero
    sin_over_norm = np.ones_like(norms)
    mask = norms > 1e-12
    sin_over_norm[mask] = sin_norms[mask] / norms[mask]
    
    # Result = cos_norms[:, None] * X + sin_over_norm[:, None] * W
    return cos_norms[:, None] * X + sin_over_norm[:, None] * W

def _parallel_transport(X_old, X_new, V_old):
    # Transport V_old from X_old to X_new
    # P(V) = V + (X_old - X_new) (V . X_old)
    dot = np.sum(V_old * X_old, axis=1)
    return V_old + (X_old - X_new) * dot[:, None]

def _line_search(X, G, D, energy, max_iter=20):
    # Backtracking line search
    # Find t such that energy(exp_X(tD)) < energy - c * t * <G, D>
    # c = 0.1 (Armijo)
    c = 0.1
    t = 1.0
    G_D = np.sum(G * D)
    for _ in range(max_iter):
        X_new = _exp_map(X, D, t)
        e_new, _ = _energy_and_grad(X_new)
        if e_new < energy + c * t * G_D:
            return X_new, e_new, t
        t *= 0.5
    return X, energy, t

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_X = None
    best_E = float('inf')
    
    inits = [_fibonacci_init(), _halton_init()]
    # Add a random init?
    # np.random.seed(42)
    # inits.append(np.random.randn(N, 3))
    # inits[-1] /= np.linalg.norm(inits[-1], axis=1)[:, None]
    
    for X0 in inits:
        X = X0
        E, G = _energy_and_grad(X)
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
            
        # Riemannian Conjugate Gradient
        D = -G
        p = D
        G_old = G
        E_old = E
        
        max_rcg_iter = 100
        for k in range(max_rcg_iter):
            if time.time() > deadline:
                break
            
            # Line search
            X_new, E_new, t = _line_search(X, G, D, E)
            
            if not np.isfinite(E_new) or E_new >= E:
                # If line search fails or doesn't improve, break
                break
            
            # Update X, E, G
            G_new, _ = _energy_and_grad(X_new) # We already computed this in line search? No, _line_search returns X_new, E_new, t. It computes G_new internally but doesn't return it. Let's modify _line_search or just recompute. Recomputing is cheap.
            
            # Parallel transport D
            D_new = _parallel_transport(X, X_new, D)
            
            # Beta
            G_norm_sq = np.sum(G_new * G_new)
            G_old_norm_sq = np.sum(G * G)
            
            if G_old_norm_sq < 1e-12:
                beta = 0.0
            else:
                # Fletcher-Reeves
                beta = G_norm_sq / G_old_norm_sq
                # Polyak might be more robust
                # beta = np.sum(G_new * G_new) / np.sum(G * G)
                
            D_new = -G_new + beta * D_new
            
            # Restart if necessary (if D_new and G_new are not sufficiently orthogonal)
            if np.sum(D_new * G_new) > 0:
                D_new = -G_new
                beta = 0.0
            
            X, E, G = X_new, E_new, G_new
            D = D_new
            
            if E < best_E:
                best_E = E
                best_X = X.copy()
                
            # Check convergence
            if np.linalg.norm(G, ord=np.inf) < 1e-8:
                break
                
    return best_X
