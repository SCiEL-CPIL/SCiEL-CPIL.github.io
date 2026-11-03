import numpy as np
import time

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
        
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _generate_bcc_init(n):
    """
    Generate an initial configuration based on a Body-Centered Cubic (BCC) lattice.
    """
    # BCC lattice vectors for a 3x3x3 supercell
    a = 1.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    # Scale to fit in [-0.45, 0.45] to allow some wall breathing room
    max_val = np.max(np.abs(points))
    points *= (0.45 / max_val)
    
    # We have 54 points. We need 50.
    idx = np.random.permutation(len(points))
    points = points[idx[:n]]
    
    return points

def _optimize_sa(points, max_time):
    """
    Simulated Annealing to maximize the minimum radius.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.02 # Initial temperature
    T_min = 0.0005
    cooling_rate = 0.9995
    step_size = 0.05
    
    while time.time() - start_time < max_time:
        if T < T_min:
            T = 0.02
            step_size = 0.05
            
        # Propose a move
        idx = np.random.randint(N)
        delta = np.random.randn(3) * step_size
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, -0.5, 0.5)
        
        new_score = _score(new_points)
        
        # Acceptance
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta_score = current_score - new_score
            if delta_score < 1e-10:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_score / T)
            
            if np.random.random() < accept_prob:
                current_points = new_points
                current_score = new_score
        
        T *= cooling_rate
        step_size *= 0.999
        
    return best_points, best_score

def _polish(points, max_time):
    """
    Local optimization using gradient ascent on the minimum distance.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    lr = 0.01
    decay = 0.999
    
    while time.time() - start_time < max_time:
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        wall_dists = 0.5 - np.abs(current_points)
        min_wall_idx = np.argmin(np.min(wall_dists, axis=1))
        min_wall_val = np.min(wall_dists[min_wall_idx])
        
        min_pair_val = dists[i, j] / 2.0
        
        if min_pair_val < min_wall_val:
            direction = current_points[i] - current_points[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                current_points[i] += lr * direction
                current_points[j] -= lr * direction
        else:
            close_wall = np.argmin(np.abs(current_points[min_wall_idx]))
            sign = np.sign(current_points[min_wall_idx, close_wall])
            if sign == 0: sign = 1
            current_points[min_wall_idx, close_wall] += lr * sign
        
        current_points = np.clip(current_points, -0.5, 0.5)
        
        new_score = _score(current_points)
        if new_score > best_score:
            best_score = new_score
            best_points = current_points.copy()
            
        lr *= decay
        if lr < 1e-6:
            break
            
    return best_points, best_score

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Run multiple restarts
    while time.time() - start_time < DEADLINE_SECONDS:
        # Initialize with BCC
        points = _generate_bcc_init(N)
        points += np.random.randn(N, 3) * 0.005
        points = np.clip(points, -0.5, 0.5)
        
        # Optimize: SA then Polish
        opt_points, opt_score = _optimize_sa(points, 5.0)
        opt_points, opt_score = _polish(opt_points, 2.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
            
    if best_points is None:
        best_points = _generate_bcc_init(N)
        
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
