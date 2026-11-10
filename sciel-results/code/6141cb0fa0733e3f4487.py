"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = 120.0  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the maximum common radius for the given centres."""
    n = len(pts)
    # Wall distance
    wall = np.minimum(0.5 - np.abs(pts), 0.5)  # min over x,y coords
    r_wall = np.min(wall)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return min(r_wall, r_pair)


def _initial_hex(N):
    """Generate a hexagonal lattice packing initialisation."""
    # Hexagonal lattice: rows offset by half the horizontal spacing
    # For N=50, use roughly 7x8 arrangement with hex offset
    # Horizontal spacing s, vertical spacing s*sqrt(3)/2
    # We want to fit in [-0.5, 0.5]^2
    
    # Try different grid sizes and pick the best initial radius
    best_pts = None
    best_r = 0.0
    
    for rows in range(5, 10):
        for cols in range(5, 10):
            if rows * cols < N:
                continue
            # Hex lattice: horizontal spacing = 2r, vertical spacing = 2r*sqrt(3)/2 = r*sqrt(3)
            # For a hex lattice with spacing s (distance between nearest neighbours):
            # horizontal: s, vertical: s*sqrt(3)/2
            # Width needed: (cols-1)*s + s/2 (for offset) if odd rows
            # Actually for hex: row 0 at x=0,s,2s,...; row 1 at x=s/2, 3s/2,...
            
            s = 1.0  # will scale
            # Compute extent
            # Max x extent: (cols-1)*s for even rows, (cols-1)*s + s/2 for odd
            max_x = (cols - 1) * s
            max_y = (rows - 1) * s * math.sqrt(3) / 2.0
            
            # Scale to fit in [-0.5, 0.5]
            scale = 1.0 / max(max_x, max_y) if max(max_x, max_y) > 0 else 1.0
            
            pts = []
            for row in range(rows):
                for col in range(cols):
                    x = col * s + (s / 2.0 if row % 2 == 1 else 0.0)
                    y = row * s * math.sqrt(3) / 2.0
                    # Center the lattice
                    x -= max_x / 2.0
                    y -= max_y / 2.0
                    # Scale
                    x *= scale
                    y *= scale
                    pts.append([x, y])
            
            pts = np.array(pts[:N])
            if len(pts) < N:
                continue
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts


def _simulated_annealing(pts, t_start, t_end, n_iter, deadline):
    """Refine points using simulated annealing."""
    n = len(pts)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Temperature
        t = t_start * (t_end / t_start) ** (i / max(n_iter - 1, 1))
        
        # Random perturbation
        candidate = current + np.random.randn(n, 2) * t * 0.05
        
        # Keep in bounds
        candidate = np.clip(candidate, -0.49, 0.49)
        
        cand_r = _compute_radius(candidate)
        
        if cand_r >= current_r or np.random.random() < math.exp((cand_r - current_r) / max(t, 1e-10)):
            current = candidate
            current_r = cand_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
    
    return best, best_r


def _local_refinement(pts, deadline):
    """Do a few iterations of gradient-like local refinement."""
    n = len(pts)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    step = 0.005
    for it in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    trial = current.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    if np.any(np.abs(trial) > 0.49):
                        continue
                    trial_r = _compute_radius(trial)
                    if trial_r > current_r:
                        current = trial
                        current_r = trial_r
                        improved = True
                        break
                if improved:
                    break
            if improved:
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
                break
        
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    
    return best, best_r


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - 5.0  # 5s safety margin
    
    # Start with hexagonal lattice
    pts = _initial_hex(N)
    best = pts.copy()
    best_r = _compute_radius(best)
    
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    
    # Phase 1: SA from hex init
    if time.time() < deadline:
        sa_pts, sa_r = _simulated_annealing(best, 0.5, 0.01, 5000, deadline)
        if sa_r > best_r:
            best = sa_pts
            best_r = sa_r
    
    # Phase 2: Local refinement
    if time.time() < deadline:
        loc_pts, loc_r = _local_refinement(best, deadline)
        if loc_r > best_r:
            best = loc_pts
            best_r = loc_r
    
    # Phase 3: More SA with smaller steps for polish
    if time.time() < deadline:
        sa2_pts, sa2_r = _simulated_annealing(best, 0.05, 0.001, 3000, deadline)
        if sa2_r > best_r:
            best = sa2_pts
            best_r = sa2_r
    
    # Phase 4: Final local refinement
    if time.time() < deadline:
        loc2_pts, loc2_r = _local_refinement(best, deadline)
        if loc2_r > best_r:
            best = loc2_pts
            best_r = loc2_r
    
    # Ensure validity: clip to bounds
    best = np.clip(best, -0.5, 0.5)
    
    return best
# EVOLVE-BLOCK-END
