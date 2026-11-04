import numpy as np
from scipy.optimize import differential_evolution
import time

N = 12

def get_all_vertices(centers, ys, thetas):
    thetas_rad = np.deg2rad(thetas)
    angles = thetas_rad[:, np.newaxis] + np.arange(6) * np.pi / 3
    x = centers[:, np.newaxis] + np.cos(angles)
    y = ys[:, np.newaxis] + np.sin(angles)
    return np.stack((x, y), axis=-1)

def fitness(params):
    inner_params = params[:N*3]
    phi = params[N*3]
    
    centers = inner_params[0::3]
    ys = inner_params[1::3]
    thetas = inner_params[2::3]
    
    vertices = get_all_vertices(centers, ys, thetas)
    
    phi_rad = np.deg2rad(phi)
    normal_angles = phi_rad + np.arange(6) * np.pi / 3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    projections = np.einsum('ijk,kl->ijl', vertices, normals)
    max_proj = np.max(projections)
    
    S_req = (2.0 / np.sqrt(3.0)) * max_proj
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    violations = np.maximum(0, 2.0 - dists)
    penalty = np.sum(violations ** 2)
    
    return S_req + penalty

def construct():
    start_time = time.time()
    
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 360.0))
    bounds.append((0.0, 60.0))
    
    try:
        result = differential_evolution(
            fitness, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=50,
            tol=1e-8,
            polish=False
        )
    except Exception:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    elapsed = time.time() - start_time
    if elapsed > 120:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_params = result.x
    best_score = result.fun
    
    inner_params = best_params[:N*3]
    phi = best_params[N*3]
    
    centers = inner_params[0::3]
    ys = inner_params[1::3]
    thetas = inner_params[2::3]
    
    # Recalculate S
    vertices = get_all_vertices(centers, ys, thetas)
    phi_rad = np.deg2rad(phi)
    normal_angles = phi_rad + np.arange(6) * np.pi / 3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    projections = np.einsum('ijk,kl->ijl', vertices, normals)
    max_proj = np.max(projections)
    S_final = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Check validity
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    if np.min(dists) < 2.0:
        # If overlap, try to use a simple valid configuration
        # Or just return it and hope the penalty was high enough that DE avoided it?
        # No, if min dist < 2.0, it's likely invalid.
        # Let's return the seed if invalid.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        inner.append([centers[i], ys[i], thetas[i]])
        
    return inner, (0.0, 0.0), S_final, phi
