import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(X):
    # X is (N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    P = X / norms
    
    # Pairwise differences
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0)
    
    # Energy
    # Sum over i < j
    triu_indices = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[triu_indices]))
    
    # Gradient w.r.t P
    # G_P[i] = - sum_{j != i} (P_i - P_j) / dist_ij^2
    # Note: dists[i,j] is dist between i and j
    # We need to sum over j for each i
    # diff[i,j] = P_i - P_j
    # inv_d2 = 1 / dists^2
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    
    # G_P = - sum_j diff[i,j] * inv_d2[i,j]
    # diff is (N, N, 3)
    # inv_d2 is (N, N)
    G_P = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    # G_tan = G_P - (G_P . P) P
    dot = np.sum(G_P * P, axis=1, keepdims=True)
    G_tan = G_P - dot * P
    
    # Backprop to X
    # G_X = G_tan / norms
    G_X = G_tan / norms
    
    return energy, G_X.flatten()

def _optimize_single(x0_flat, time_limit):
    start_time = time.time()
    
    # L-BFGS-B is good for large N
    # We need to stop if time runs out
    # scipy minimize doesn't have a time limit callback directly, 
    # but we can check in the callback
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit reached")
        return False
        
    try:
        res = minimize(
            _energy_and_grad, 
            x0_flat, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-8, 'disp': False, 'maxls': 20}
        )
        # Check if it converged or just stopped
        # We always return the best point found by the optimizer
        return res.x, res.fun
    except Exception:
        # If time limit raised, res is not available. 
        # We need to keep track of best point manually or use a wrapper.
        # Actually, if exception is raised, we lose the state.
        # Better to use a callback that sets a flag and returns True to stop.
        pass
    
    # The above try-except is flawed for time limits because minimize might not catch the exception cleanly to return the last state.
    # Let's use a callback that returns True to stop.
    
    stop_flag = [False]
    def time_callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
            return True
        return False

    try:
        res = minimize(
            _energy_and_grad, 
            x0_flat, 
            method='L-BFGS-B', 
            jac=True,
            callback=time_callback,
            options={'maxiter': 1000, 'gtol': 1e-8, 'disp': False}
        )
        return res.x, res.fun
    except Exception as e:
        # Fallback to x0 if something goes wrong
        return x0_flat, _energy_and_grad(x0_flat.reshape(N, 3))[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initializations
    inits = []
    fib = _fibonacci_sphere().flatten()
    inits.append(fib)
    
    # Add random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random points on sphere
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v.flatten())
        
    # Add perturbed fib
    for _ in range(5):
        noise = rng.normal(size=(N, 3)) * 0.1
        p = fib.reshape(N, 3) + noise
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        inits.append(p.flatten())
    
    best_x = None
    best_energy = float('inf')
    
    n_inits = len(inits)
    time_per_init = (deadline - time.time()) / n_inits
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        alloc_time = min(time_per_init, remaining * 0.9) # Leave some time
        
        x_opt, e_opt = _optimize_single(x0, alloc_time)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
    
    if best_x is None:
        # Fallback
        best_x = fib
        best_energy = _energy_and_grad(fib.reshape(N, 3))[0]
        
    # Final polish on the best solution?
    # We have some time left maybe.
    if time.time() < deadline - 10:
        time_left = deadline - time.time() - 5
        x_opt, e_opt = _optimize_single(best_x, time_left)
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
            
    # Normalize final result
    best_pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
