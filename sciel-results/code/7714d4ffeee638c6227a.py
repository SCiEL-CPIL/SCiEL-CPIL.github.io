"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    # Start with grid
    c = _init_grid()
    best_c = c.copy()
    best_score = _score(c)

    # Also try a few random starts
    for _ in range(5):
        if time.time() > deadline:
            break
        cr = _init_random()
        sr = _score(cr)
        if sr > best_score:
            best_score = sr
            best_c = cr.copy()

    # Local optimization using steepest ascent / gradient-like method
    # We'll use a simple iterative refinement: for each sphere, move it slightly
    # in the direction that increases the minimum distance to neighbors and walls.
    
    max_iters = 500
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Compute gradient-like direction for sphere i
            # Force from walls: push away from nearest wall
            grad = np.zeros(3)
            for k in range(3):
                if c[i, k] > 0:
                    # closer to +x wall
                    wall_dist = 0.5 - c[i, k]
                    grad[k] -= 1.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = c[i, k] + 0.5
                    grad[k] += 1.0 / (wall_dist + 1e-6)
            
            # Force from other spheres
            for j in range(N):
                if i == j:
                    continue
                d = c[i] - c[j]
                dist = np.sqrt(np.sum(d**2))
                if dist < 1e-10:
                    # random nudge
                    d = np.random.randn(3)
                    dist = np.sqrt(np.sum(d**2))
                grad += d / (dist**2)
            
            # Take a step
            new_c = c.copy()
            new_c[i] = c[i] + lr * grad
            # Clamp to cube
            new_c[i] = np.clip(new_c[i], -0.499, 0.499)
            
            new_score = _score(new_c)
            if new_score > _score(c) + 1e-9:
                c = new_c
                improved = True
        
        if not improved:
            break
        
        if _score(c) > best_score:
            best_score = _score(c)
            best_c = c.copy()
    
    # Try a few random restarts with local opt if time permits
    for restart in range(3):
        if time.time() > deadline:
            break
        
        c2 = _init_random()
        # Local opt on c2
        for it in range(200):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                grad = np.zeros(3)
                for k in range(3):
                    if c2[i, k] > 0:
                        wall_dist = 0.5 - c2[i, k]
                        grad[k] -= 1.0 / (wall_dist + 1e-6)
                    else:
                        wall_dist = c2[i, k] + 0.5
                        grad[k] += 1.0 / (wall_dist + 1e-6)
                
                for j in range(N):
                    if i == j:
                        continue
                    d = c2[i] - c2[j]
                    dist = np.sqrt(np.sum(d**2))
                    if dist < 1e-10:
                        d = np.random.randn(3)
                        dist = np.sqrt(np.sum(d**2))
                    grad += d / (dist**2)
                
                new_c2 = c2.copy()
                new_c2[i] = c2[i] + lr * grad
                new_c2[i] = np.clip(new_c2[i], -0.499, 0.499)
                
                if _score(new_c2) > _score(c2) + 1e-9:
                    c2 = new_c2
                    improved = True
            
            if not improved:
                break
        
        s2 = _score(c2)
        if s2 > best_score:
            best_score = s2
            best_c = c2.copy()
    
    return best_c
# EVOLVE-BLOCK-END
