import numpy as np
import time

N = 100
DEADLINE = 145.0  # Safety margin for 150s limit

def _score(c):
    """Compute the largest common radius for centres c (shape (N, 3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)

def _init_grid():
    """
    Initialize with a 4x5x5 grid. 
    4*5*5 = 100 points exactly.
    This provides a structured starting point.
    """
    nx, ny, nz = 4, 5, 5
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _compute_knn(c, k=6):
    """
    Compute the k-nearest neighbors for each point in c.
    Returns a list of lists of indices.
    """
    # Using scipy.spatial.cKDTree for efficiency if available, 
    # but to stick to stdlib+numpy+scipy optimize, we can use scipy.spatial
    from scipy.spatial import cKDTree
    tree = cKDTree(c)
    # k+1 because it includes the point itself
    _, indices = tree.query(c, k=k+1)
    # Remove self (index 0 in each row is self)
    neighbors = [list(ind[1:]) for ind in indices]
    return neighbors

def construct():
    start_time = time.time()
    
    # 1. Initial Grid
    c = _init_grid()
    r_current = _score(c)
    
    # 2. Simulated Annealing / Local Search
    # We aim to maximize the minimum clearance.
    # Objective: max min( min_pair_dist/2, min_wall_dist )
    
    # Pre-compute KNN to guide moves (topological/structural moves)
    # We will update KNN periodically because it's expensive to update every step.
    neighbors = _compute_knn(c)
    
    best_c = c.copy()
    best_r = r_current
    
    # Annealing parameters
    T_start = 0.01
    T_end = 1e-5
    alpha = 0.999 # Cooling rate
    T = T_start
    
    # We will perform moves until time runs out or T is very low.
    # Moves:
    # 1. Random Perturbation: Pick random i, add small noise.
    # 2. Swap: Pick random i, j (prefer neighbors), swap positions? 
    #    Actually, swapping positions of spheres i and j is equivalent to moving i to j's pos and j to i's.
    #    Since spheres are identical, swapping labels doesn't change the set of centers.
    #    So "Swap" isn't useful unless we mean "Exchange positions with a neighbor".
    #    Better move: Move sphere i towards/away from sphere j.
    #    Or: Move sphere i to a random position within a small sphere around its current pos.
    #    Or: "Topological Sort" idea: Iterate through spheres in a sorted order (e.g., by x-coord) 
    #    and perform local adjustments.
    
    # Let's use a hybrid:
    # 1. Random local move (Metropolis step).
    # 2. Periodic "Sorting/Re-indexing" to maintain structural coherence?
    #    Actually, the positions are what matter.
    
    # Optimization Loop
    step = 0
    last_knn_update = 0
    
    while time.time() < DEADLINE:
        step += 1
        
        # Update KNN every 500 steps
        if step - last_knn_update > 500:
            neighbors = _compute_knn(c)
            last_knn_update = step
        
        # Choose a move
        # 80% chance: Random Perturbation
        # 20% chance: Guided Move (towards/away from a neighbor)
        
        i = np.random.randint(N)
        move_type = np.random.random()
        
        if move_type < 0.8:
            # Random Perturbation
            # Scale perturbation by T
            noise_scale = 0.01 * (T / T_start + 0.1)
            delta = np.random.normal(0, noise_scale, size=3)
            new_c = c.copy()
            new_c[i] += delta
        else:
            # Guided Move: Move i relative to a neighbor j
            if neighbors[i]:
                j = neighbors[i][np.random.randint(len(neighbors[i]))]
            else:
                j = np.random.randint(N)
                if j == i: continue
                
            # Move i slightly towards or away from j
            vec = c[j] - c[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue
            unit_vec = vec / dist
            
            # Random step size
            step_size = np.random.uniform(0.005, 0.02)
            # Random direction: towards or away
            direction = 1.0 if np.random.random() < 0.5 else -1.0
            
            delta = direction * step_size * unit_vec
            new_c = c.copy()
            new_c[i] += delta
        
        # Clip to box
        new_c = np.clip(new_c, -0.5, 0.5)
        
        # Evaluate
        new_r = _score(new_c)
        
        # Metropolis Acceptance
        if new_r > best_r:
            c = new_c
            best_c = new_c.copy()
            best_r = new_r
        else:
            # Accept if exp((new_r - current_r) / T) > random
            # Note: We are maximizing. 
            # delta_r = new_r - _score(c)
            # If new_r < current_r, we might accept to escape local min.
            current_r = _score(c)
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    c = new_c
            else:
                # If new_r >= current_r but < best_r, always accept (greedy improvement)
                c = new_c
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Safety check
        if step % 1000 == 0:
             # Check time
             if time.time() > DEADLINE - 5:
                 break

    return best_c

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Radius: {_score(res):.10f}")
    print(f"Shape: {res.shape}")
