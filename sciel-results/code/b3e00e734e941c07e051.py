import numpy as np
from scipy.optimize import minimize
import time

N = 32

def sum_radii(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii of the Voronoi cells clipped to [0,1]^2
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p) # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1) # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def grad_sum_radii(c):
    """
    Gradient of sum_radii using subgradient method for non-smooth points
    """
    p = c.reshape(N, 2)
    grad = np.zeros_like(p)
    
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    # Distances to other centers
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Find closest neighbor for each point
    min_neighbor_dist = np.min(dists, axis=1)
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        if wall_r < neigh_r - 1e-10:
            # Limited by wall - use subgradient (average of active constraints)
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            if active:
                avg_grad = np.mean(active, axis=0)
                grad[i] += avg_grad
        elif neigh_r < wall_r - 1e-10:
            # Limited by neighbor j
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                # r_i = 0.5 * ||p_i - p_j||
                # grad r_i wrt p_i = 0.5 * (p_i - p_j) / d
                grad[i] += 0.5 * vec / d
                # Gradient contribution to p_j from r_i
                grad[j] += -0.5 * vec / d
        else:
            # At a kink point - use average of both gradients
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active_walls = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            wall_grad = np.mean(active_walls, axis=0) if active_walls else np.zeros(2)
            
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                neigh_grad_i = 0.5 * vec / d
                neigh_grad_j = -0.5 * vec / d
            else:
                neigh_grad_i = np.zeros(2)
                neigh_grad_j = np.zeros(2)
            
            # Average of wall and neighbor gradients
            grad[i] += 0.5 * (wall_grad + neigh_grad_i)
            grad[j] += 0.5 * neigh_grad_j
    
    return grad.flatten()

def simulated_annealing(c_init, max_iter=5000, t0=1.0, t_end=1e-4, beta=0.999):
    """
    Simulated Annealing to find a good local minimum for sum_radii.
    """
    np.random.seed(0) # For reproducibility in the function, though global seed is set in construct
    c = c_init.copy()
    p = c.reshape(N, 2)
    current_energy = -sum_radii(c) # We minimize -sum_radii, so maximize sum_radii
    best_c = c.copy()
    best_energy = current_energy
    
    t = t0
    for i in range(max_iter):
        # Pick a random circle to move
        idx = np.random.randint(N)
        # Gaussian perturbation
        delta = np.random.normal(0, 0.05, 2)
        new_p = p.copy()
        new_p[idx] += delta
        # Clip to bounds
        new_p[idx] = np.clip(new_p[idx], 0.01, 0.99)
        
        new_c = new_p.flatten()
        new_energy = -sum_radii(new_c)
        
        # Acceptance probability
        if new_energy < current_energy:
            p = new_p
            c = new_c
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_c = c.copy()
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0:
                delta_e = 0
            if np.random.uniform(0, 1) < np.exp(-delta_e / t):
                p = new_p
                c = new_c
                current_energy = new_energy
        
        t *= beta
        if t < t_end:
            break
            
    return best_c

def local_polish(c, max_iter=200):
    """
    Refine using L-BFGS-B
    """
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    try:
        res = minimize(lambda c: -sum_radii(c), c, jac=lambda c: -grad_sum_radii(c),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x
    except:
        return c

def construct():
    start_time = time.time()
    timeout = 110
    
    best_c = None
    best_val = -np.inf
    
    # Initializations
    inits = []
    
    # 1. Grid
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 6)
    grid = np.array([[x, y] for y in ys for x in xs][:N]).flatten()
    inits.append(grid)
    
    # 2. Hexagonal
    hex_points = []
    for row in range(6):
        for col in range(6):
            if len(hex_points) < N:
                x = (col + 0.5 + (row % 2) * 0.5) / 6.0
                y = (row + 0.5) / 6.0
                hex_points.append([x, y])
    inits.append(np.array(hex_points[:N]).flatten())
    
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        p = np.random.uniform(0.1, 0.9, size=(N, 2))
        inits.append(p.flatten())
        
    # 4. Halton
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    halton_points = []
    for i in range(1, N + 1):
        halton_points.append([halton(i, 2), halton(i, 3)])
    inits.append(np.array(halton_points).flatten())

    # Run SA for each init
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
            
        try:
            # SA
            c_sa = simulated_annealing(c_init, max_iter=2000)
            val = sum_radii(c_sa)
            if val > best_val:
                best_val = val
                best_c = c_sa.copy()
                
            # Polish
            if time.time() - start_time < timeout - 10:
                c_polished = local_polish(c_sa, max_iter=100)
                val = sum_radii(c_polished)
                if val > best_val:
                    best_val = val
                    best_c = c_polished.copy()
        except:
            pass

    # Final global polish on the best candidate
    if best_c is not None:
        if time.time() - start_time < timeout - 5:
            try:
                # One more round of SA with lower temp
                c_final_sa = simulated_annealing(best_c, max_iter=5000, t0=0.1, t_end=1e-5, beta=0.999)
                val = sum_radii(c_final_sa)
                if val > best_val:
                    best_val = val
                    best_c = c_final_sa.copy()
                    
                # Final L-BFGS-B
                if time.time() - start_time < timeout - 1:
                    c_final = local_polish(best_c, max_iter=300)
                    val = sum_radii(c_final)
                    if val > best_val:
                        best_val = val
                        best_c = c_final.copy()
            except:
                pass

    if best_c is None:
        best_c = inits[0]
        best_val = sum_radii(best_c)
        
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
