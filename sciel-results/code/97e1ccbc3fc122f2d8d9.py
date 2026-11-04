import numpy as np
import time

N = 150

def fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_points(rng):
    v = rng.normal(size=(N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def compute_energy_grad(X):
    # X is (N, 3)
    # Compute squared distances
    # D2[i, j] = ||x_i - x_j||^2
    # = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    # Since ||x_i|| = 1, D2[i, j] = 2 - 2 x_i . x_j
    X2 = np.sum(X * X, axis=1) # should be 1
    D2 = X2[:, None] + X2[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(D2, 1.0) # Avoid 0/0, will be masked
    D = np.sqrt(np.maximum(D2, 1e-12))
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    # Sum over i < j
    # We can sum over all i != j and divide by 2
    E = -0.5 * np.sum(np.log(D[mask]))
    
    # Gradient
    # G_i = sum_{j != i} (x_i - x_j) / D_ij^2
    # W_ij = 1 / D_ij^2
    W = 1.0 / D2
    np.fill_diagonal(W, 0.0)
    
    # G = X * sum(W, axis=1) - X @ W
    # sum(W, axis=1) is (N, 1)
    # X * (N, 1) is (N, 3)
    # X @ W is (N, 3)
    G = X * W.sum(axis=1, keepdims=True) - X @ W
    
    return E, G

def optimize(X_init, time_limit, seed=0):
    X = X_init.copy()
    best_E, best_X = np.inf, X.copy()
    
    # Initial gradient
    E, G = compute_energy_grad(X)
    best_E = E
    best_X = X.copy()
    
    prev_X = X.copy()
    prev_G = G.copy()
    
    step_size = 1e-3
    bb_step = step_size
    
    rng = np.random.default_rng(seed)
    
    for iter in range(5000):
        if time.time() > time_limit:
            break
        
        # Tangent gradient
        G_tan = G - (G * X).sum(axis=1, keepdims=True) * X
        
        # Check convergence
        if np.linalg.norm(G_tan) < 1e-8:
            break
            
        # BB Step Size
        s = X - prev_X
        y = G - prev_G
        sy = np.sum(s * y)
        ss = np.sum(s * s)
        
        if sy > 1e-10:
            bb_step = ss / sy
            bb_step = np.clip(bb_step, 1e-6, 1e-2)
        else:
            # Fall back
            bb_step = step_size * 0.5
            
        step_size = bb_step
        
        # Update
        X_new = X + step_size * G_tan
        X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
        
        E_new, G_new = compute_energy_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            prev_X = X.copy()
            prev_G = G_new.copy()
            
            if E < best_E:
                best_E = E
                best_X = X.copy()
        else:
            # Backtracking
            step_size *= 0.5
            if step_size < 1e-8:
                # Stuck
                # Try a small random jitter
                X = X + rng.normal(scale=1e-4, size=(N, 3))
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                E, G = compute_energy_grad(X)
                prev_X = X.copy()
                prev_G = G.copy()
                step_size = 1e-4
            else:
                # Retry with smaller step
                X_new = X + step_size * G_tan
                X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
                E_new, G_new = compute_energy_grad(X_new)
                if E_new < E:
                    X = X_new
                    E = E_new
                    prev_X = X.copy()
                    prev_G = G_new.copy()
                    if E < best_E:
                        best_E = E
                        best_X = X.copy()
                else:
                    # Still failing, jitter and reset
                    X = X + rng.normal(scale=1e-4, size=(N, 3))
                    X = X / np.linalg.norm(X, axis=1, keepdims=True)
                    E, G = compute_energy_grad(X)
                    prev_X = X.copy()
                    prev_G = G.copy()
                    step_size = 1e-4

    return best_X, best_E

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_energy = np.inf
    best_points = None
    
    # Initializations
    inits = [fibonacci_points()]
    rng = np.random.default_rng(42)
    for i in range(4):
        inits.append(random_points(rng))
    
    for i, X_init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        X_opt, E_opt = optimize(X_init, deadline, seed=100+i)
        if E_opt < best_energy:
            best_energy = E_opt
            best_points = X_opt
            
    # Final validation
    if best_points is None:
        return fibonacci_points()
        
    # Check for close points
    D2 = np.sum(best_points**2, axis=1)[:,None] + np.sum(best_points**2, axis=1)[None,:] - 2 * (best_points @ best_points.T)
    np.fill_diagonal(D2, 1.0)
    min_dist = np.sqrt(np.min(D2))
    
    if min_dist < 1e-9:
        # Perturb
        rng = np.random.default_rng(999)
        best_points = best_points + rng.normal(scale=1e-5, size=(N, 3))
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        
    return best_points
