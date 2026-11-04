"""Packing 100 equal spheres in a unit cube: multi-restart vectorized simulated annealing."""
import time
import numpy as np
from scipy.spatial.distance import pdist

N = 100
HALF = 0.5
DEADLINE = 140.0  # safety margin before 150s kill

def _score(pts):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -HALF - 1e-12) or np.any(pts > HALF + 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(pts)
    min_pair = np.min(d) if len(d) > 0 else 1.0
    # Distance to walls
    wall_dists = np.minimum(np.abs(pts) + HALF, HALF - np.abs(pts))
    # Actually: distance from center to nearest wall
    # For a point at coord c in [-0.5, 0.5], distance to nearest wall is min(0.5-|c|)
    wall = np.min(HALF - np.abs(pts))
    return min(0.5 * min_pair, wall)

def _random_init():
    """Random initialization within the cube."""
    return np.random.uniform(-HALF, HALF, size=(N, 3))

def _lattice_init():
    """
    Hybrid initialization: 64 spheres in a 4x4x4 grid, 36 spheres in interstitial positions.
    This gives a geometrically structured starting point.
    """
    # 4x4x4 grid: 64 points
    m = 4
    # Scale so that 64 points fit nicely; use spacing to maximize radius
    # In a 4x4x4 grid in [-0.5, 0.5], spacing = 1/(m-1) = 1/3
    # But we want the grid to be centered. Let's use ticks at -0.5+0.5*(i+0.5)/m
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # That's 64 points. Spacing between adjacent = 1/4 = 0.25, so radius from pair = 0.125
    # Wall distance = 0.5 - max|coord| = 0.5 - 0.375 = 0.125
    # So r = 0.125 for the grid alone.
    
    # Now add 36 more spheres in interstitial voids.
    # The interstitial positions in a simple cubic lattice are at the body centers of the cubes.
    # For a 4x4x4 grid, the interstitial sites are at the centers of the 3x3x3 = 27 small cubes.
    # But we need 36, so we'll also use face-centered positions.
    # Actually, let's just place 36 additional spheres at randomized positions near the interstitial sites,
    # or use a different strategy: place them at positions that are shifted from the grid.
    
    # Alternative: use a 5x5x5 grid but only take 100 points optimally.
    # Let's try: 5x5x5 = 125 points, take 100.
    # Spacing = 1/4 = 0.25, radius = 0.125. But we only use 100 of 125.
    
    # Better: use the 4x4x4 grid (64 pts) and add 36 pts at face-center interstitials.
    # Face centers of the 3x3x3 sub-cubes: each sub-cube has 6 faces, but shared faces.
    # Let's just randomly perturb some positions or use a second lattice.
    
    # For now, let's use a simpler approach: 5x5x5 grid, take first 100.
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
    # 125 points, take 100
    return grid2[:N]

def _sa_optimize(pts0, t_start, max_steps=500000):
    """
    Vectorized simulated annealing.
    Returns the best point configuration found.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Initial temperature
    r0 = best_score
    T = max(0.001, r0 * 0.1)
    T_min = 1e-7
    
    # Step size
    step_scale = max(0.01, r0 * 0.5)
    
    rng = np.random.RandomState()
    
    for step in range(max_steps):
        # Deadline check
        if time.time() - t_start > DEADLINE:
            break
        
        # Random perturbation: move a random subset of points
        n_move = rng.randint(1, 6)
        idx = rng.choice(N, size=n_move, replace=False)
        
        old_positions = pts[idx].copy()
        pts[idx] += rng.normal(0, step_scale, size=(n_move, 3))
        
        # Clip to cube (soft: allow slight overshoot but score will penalize)
        # Actually, let's not clip; the score function handles it by returning 0 if out of bounds.
        # But that's too harsh. Let's clip to be safe.
        pts[idx] = np.clip(pts[idx], -HALF, HALF)
        
        new_score = _score(pts)
        
        # Acceptance criterion
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
            # Cool down slowly when improving
            T *= 0.999
            step_scale *= 0.999
        else:
            # Metropolis acceptance
            delta = new_score - _score(old_positions)
            # We need the old score; let's compute it properly
            # Actually, we should track current score
            # Let's restructure: keep track of current score
            
            # For simplicity, use a simpler acceptance
            accept_prob = np.exp((new_score - best_score) / max(T, 1e-10))
            if rng.random() < accept_prob:
                # Accept
                pass
            else:
                # Reject
                pts[idx] = old_positions
        
        # Cool down
        T *= 0.9999
        step_scale *= 0.99999
        
        if T < T_min:
            break
    
    return best_pts, best_score

def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = []
    
    # 1. Lattice-based init
    inits.append(_lattice_init())
    
    # 2. Random inits
    for _ in range(3):
        inits.append(_random_init())
    
    # 3. Hybrid: 4x4x4 grid + perturbations
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add 36 more points at face-centered positions of the sub-cubes
    # Face centers: for each pair of adjacent grid points, the midpoint
    # Let's just use the 5x5x5 grid truncated
    inits.append(grid_pts[:64].copy())
    # Add 36 random points near interstitial sites
    interstitial = np.array([(x + 0.5/4, y + 0.5/4, z + 0.5/4) 
                              for z in ticks[:-1] for y in ticks[:-1] for x in ticks[:-1]])
    # That's 27 points. Add 9 more at face centers
    face_pts = []
    for i in range(3):
        for j in range(3):
            # Center of face between grid[i] and grid[i+1] in x, at y=j, z=k
            for k in range(3):
                face_pts.append((ticks[i] + 0.5/4, ticks[j], ticks[k]))
    face_pts = np.array(face_pts[:9])
    hybrid = np.vstack([grid_pts, interstitial, face_pts])[:N]
    inits.append(hybrid)
    
    # Run SA on each init
    for init in inits:
        if time.time() - t_start > DEADLINE - 5:
            break
        
        # Ensure valid start
        init = np.clip(init, -HALF, HALF)
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
        
        # Run SA
        pts, score = _sa_optimize(init, t_start)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final: ensure we return a valid result
    if best_pts is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts
