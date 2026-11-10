"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    # Min distance between any two centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Min distance to walls
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)

def _push_optimize(pts, max_iter=2000, deadline=None):
    """
    Direct bottleneck-targeting optimization.
    Identifies the specific constraint (pair or wall) and pushes to relieve it.
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    step = 0.02
    decay = 0.99
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Identify bottleneck
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        dmin = np.min(dists)
        
        wall_dists = 0.5 - np.abs(pts)
        wmin = np.min(wall_dists)
        
        current_r = min(dmin / 2.0, wmin)
        
        forces = np.zeros_like(pts)
        
        if dmin / 2.0 < wmin:
            # Bottleneck is a pair
            # Find the closest pair
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx
            if i > j: i, j = j, i
            
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(3) * 0.01
                dist = np.linalg.norm(d)
            
            # Push i and j apart
            force_vec = d / dist
            forces[i] += force_vec
            forces[j] -= force_vec
            
            # Also add slight repulsion from other close neighbors to avoid immediate re-collision
            for k in range(n):
                if k == i or k == j: continue
                d_ik = pts[i] - pts[k]
                dist_ik = np.linalg.norm(d_ik)
                if dist_ik < dmin * 1.5:
                    forces[i] += (d_ik / dist_ik) * 0.1
                    forces[k] -= (d_ik / dist_ik) * 0.1
                d_jk = pts[j] - pts[k]
                dist_jk = np.linalg.norm(d_jk)
                if dist_jk < dmin * 1.5:
                    forces[j] += (d_jk / dist_jk) * 0.1
                    forces[k] -= (d_jk / dist_jk) * 0.1
        else:
            # Bottleneck is a wall
            # Find the point closest to a wall
            min_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = min_idx
            
            # Push i away from wall k
            if pts[i, k] > 0:
                forces[i, k] -= 1.0
            else:
                forces[i, k] += 1.0
            
            # Also add slight repulsion from neighbors
            for j in range(n):
                if j == i: continue
                d_ij = pts[i] - pts[j]
                dist_ij = np.linalg.norm(d_ij)
                if dist_ij < dmin * 1.5:
                    forces[i] += (d_ij / dist_ij) * 0.1
                    forces[j] -= (d_ij / dist_ij) * 0.1
        
        # Normalize and update
        fnorm = np.linalg.norm(forces, axis=1, keepdims=True)
        fnorm = np.maximum(fnorm, 1e-10)
        forces = forces / fnorm
        
        new_pts = pts + step * forces
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step *= decay
            if step < 1e-8:
                break
    
    return best_pts, best_score

def _simulated_annealing(pts, max_iter=5000, deadline=None):
    """
    Simulated annealing to escape local minima.
    """
    pts = pts.copy()
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Generate neighbor
        new_pts = pts.copy()
        # Randomly pick a few points to move
        num_moves = np.random.randint(1, 4)
        indices = np.random.choice(n, num_moves, replace=False)
        moves = np.random.randn(num_moves, 3) * T
        new_pts[indices] += moves
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_score = _score(new_pts)
        delta = new_score - best_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        T = max(T * cooling, T_min)
        
        # Periodic restart from best to avoid getting stuck
        if it % 500 == 0:
            pts = best_pts.copy()
            T = 0.02
    
    return best_pts, best_score

def _generate_bcc_config():
    """Generate a BCC lattice configuration."""
    # BCC lattice points in unit cell: (0,0,0), (0.5,0.5,0.5)
    # We want 50 points.
    # A 3x3x3 BCC grid has 27 * 2 = 54 points.
    # Let's generate a 3x3x3 BCC grid centered at 0.
    points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Corner
                points.append((i * 0.5, j * 0.5, k * 0.5))
                # Body center
                points.append((i * 0.5 + 0.25, j * 0.5 + 0.25, k * 0.5 + 0.25))
    
    points = np.array(points)
    # Scale to fit in [-0.5, 0.5]
    # The extent is roughly 1.5 * 0.5 = 0.75? No.
    # i is -1, 0, 1.
    # Corner: -0.5, 0, 0.5
    # Center: -0.25, 0.25, 0.75
    # So max extent is 0.75.
    # We need to fit in [-0.5, 0.5].
    # Scale factor: 0.5 / 0.75 = 2/3
    points *= 2.0 / 3.0
    
    # Shuffle and take 50
    np.random.shuffle(points)
    p = points[:N].copy()
    p += np.random.randn(N, 3) * 0.02
    return p

def _generate_fcc_config():
    """Generate an FCC lattice configuration."""
    points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Corner
                points.append((i * 0.5, j * 0.5, k * 0.5))
                # Face centers
                points.append((i * 0.5, j * 0.5 + 0.25, k * 0.5 + 0.25))
                points.append((i * 0.5 + 0.25, j * 0.5, k * 0.5 + 0.25))
                points.append((i * 0.5 + 0.25, j * 0.5 + 0.25, k * 0.5))
    
    points = np.array(points)
    points *= 2.0 / 3.0
    np.random.shuffle(points)
    p = points[:N].copy()
    p += np.random.randn(N, 3) * 0.02
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_score = -1.0
    
    # Generate initial configurations
    configs = [
        _generate_bcc_config(),
        _generate_fcc_config(),
        np.random.uniform(-0.4, 0.4, (N, 3))
    ]
    
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # 1. Push optimization
        pts_opt1, score1 = _push_optimize(init_pts, max_iter=2000, deadline=deadline)
        
        if score1 > best_score:
            best_score = score1
            best_pts = pts_opt1.copy()
        
        # 2. Simulated annealing from the result of push
        if time.time() < deadline:
            pts_opt2, score2 = _simulated_annealing(pts_opt1, max_iter=5000, deadline=deadline)
            
            if score2 > best_score:
                best_score = score2
                best_pts = pts_opt2.copy()
        
        # 3. Final push refinement
        if time.time() < deadline:
            pts_opt3, score3 = _push_optimize(pts_opt2, max_iter=1000, deadline=deadline)
            
            if score3 > best_score:
                best_score = score3
                best_pts = pts_opt3.copy()
    
    # Fallback
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
