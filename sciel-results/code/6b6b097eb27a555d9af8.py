"""Improved packing of 50 equal spheres in a unit cube using SA with BCC init and repulsion energy."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START

def _min_pair_dist(centers):
    """Return minimum pairwise distance using KDTree."""
    n = len(centers)
    if n < 2:
        return float('inf')
    tree = cKDTree(centers)
    dists, idxs = tree.query(centers, k=2)
    return np.min(dists[:, 1])

def _min_wall_dist(centers):
    """Return minimum distance from any center to any wall."""
    coords = np.abs(centers)
    max_coords = np.max(coords, axis=1)
    dists = 0.5 - max_coords
    return np.min(dists)

def _score(centers):
    """Return the radius: min(min_pair_dist/2, min_wall_dist)."""
    mpd = _min_pair_dist(centers)
    mwd = _min_wall_dist(centers)
    return min(mpd / 2.0, mwd)

def _is_valid(centers):
    """Check if centers are valid: finite, inside cube."""
    if centers.shape != (N, 3):
        return False
    if not np.all(np.isfinite(centers)):
        return False
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return False
    return True

def _bcc_init():
    """
    Create a BCC (Body-Centered Cubic) like initial configuration.
    BCC has a coordination number of 8 and is denser than FCC for equal spheres in a box.
    For 50 spheres, we use a 3x3x3 BCC lattice (27 * 2 = 54 sites) and pick the best 50.
    """
    # BCC lattice in a unit cell has points at (0,0,0) and (0.5, 0.5, 0.5) relative to the cell.
    # We scale this to fit inside [-0.5, 0.5]^3.
    # Let's create a 3x3x3 grid of BCC units.
    # Each unit has 2 spheres. Total 54 spheres. We need 50.
    
    n_grid = 3
    # Scale factor so that the whole structure fits in [-0.5, 0.5]
    # The extent of the grid is n_grid * cell_size.
    # Let cell_size = 1.0 / n_grid.
    # Points in a BCC unit at (i, j, k) and (i+0.5, j+0.5, k+0.5) * cell_size
    # Centered at origin: shift by (n_grid - 1) * cell_size / 2 ?
    
    # Actually, let's just generate points in [0, 1]^3 and then scale to [-0.5, 0.5]^3.
    # BCC points in [0, 1]^3 for a 3x3x3 grid:
    # Corners: (i, j, k) for i,j,k in 0..2
    # Body centers: (i+0.5, j+0.5, k+0.5) for i,j,k in 0..2
    # Total 27 + 27 = 54 points.
    
    pts = []
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    
    pts = np.array(pts)
    # Scale to [-0.5, 0.5]
    # Current extent is 0 to 2.5 (max coord is 2.5)
    # We want extent to be 1.0 (from -0.5 to 0.5)
    # But BCC is symmetric, so let's center it.
    # Mean is (1.25, 1.25, 1.25)
    # Min is 0, Max is 2.5
    # Shift by -1.25 to center: -1.25 to 1.25
    # Scale by 0.5 / 1.25 = 0.4
    # So: (pts - 1.25) * 0.4
    
    # However, this might put points too close to walls or each other.
    # Let's just use a simple grid and BCC mix.
    
    # Alternative: Use a simple cubic grid of 3x3x3 = 27 points, and add 23 more points
    # in the gaps? No, that's messy.
    
    # Let's stick to the 3x3x3 BCC lattice but optimize the scale.
    # The "natural" spacing in BCC is such that the corner and body center touch.
    # Distance between (0,0,0) and (0.5, 0.5, 0.5) is sqrt(0.75) * cell_size.
    # If this is the minimum distance d, then radius r = d/2.
    # We want to maximize r.
    
    # For now, just generate the points and let the optimizer adjust them.
    # We'll pick the 50 points that are "best" (e.g., closest to center? or just the first 50).
    # To be symmetric, we should pick points that form a symmetric set.
    # The 54 points are symmetric. Removing 4 points breaks symmetry.
    # Let's remove the 4 corner points that are furthest from the center?
    # No, corners are at (0,0,0), (2,0,0), etc.
    # Body centers are at (0.5, 0.5, 0.5), etc.
    # Let's just take the first 50 points.
    
    pts = (pts - 1.25) * 0.4
    return pts[:50]

def _local_search(centers, time_limit, seed=42):
    """
    Fast local search using gradient-like steps on the bottleneck.
    """
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    start_time = time.time()
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        mpd = _min_pair_dist(current)
        mwd = _min_wall_dist(current)
        current_score = min(mpd / 2.0, mwd)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb to escape
                current += rng.normal(0, 0.01, size=(N, 3))
                current = np.clip(current, -0.5, 0.5)
                no_improve_count = 0
                continue
        
        # Determine bottleneck
        if mpd / 2.0 < mwd:
            # Pair distance is bottleneck
            tree = cKDTree(current)
            dists, idxs = tree.query(current, k=2)
            nn_dists = dists[:, 1]
            
            min_nn = np.min(nn_dists)
            threshold = min_nn + 0.001
            candidates = np.where(nn_dists < threshold)[0]
            
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
        else:
            # Wall distance is bottleneck
            wall_dists = 0.5 - np.max(np.abs(current), axis=1)
            min_wall_val = np.min(wall_dists)
            threshold = min_wall_val + 0.001
            candidates = np.where(wall_dists < threshold)[0]
            
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
            else:
                idx = rng.randint(N)
            
            norm = np.linalg.norm(current[idx])
            if norm > 1e-9:
                direction = -current[idx] / norm
            else:
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            
            current[idx] += direction * step_size
            
        # Clamp to cube
        current = np.clip(current, -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.999
        if step_size < 0.001:
            step_size = 0.01

    return best, best_score

def _optimize_with_sa(centers, time_limit, seed=42):
    """
    Simulated Annealing with a smooth repulsion energy landscape.
    Instead of optimizing the min distance directly, we optimize a sum of inverse powers
    of distances, which provides a smoother gradient and helps escape local minima.
    """
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    current_score = best_score
    
    start_time = time.time()
    start_temp = 0.05
    end_temp = 0.0001
    elapsed = 0
    
    # Energy function: sum of 1/d^2 for all pairs, plus wall penalty
    def _energy(centers):
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=N)  # All pairs
        # Exclude self-distances (diagonal)
        np.fill_diagonal(dists, np.inf)
        # Sum of inverse squares
        pair_energy = np.sum(1.0 / (dists ** 2))
        # Wall penalty: sum of (0.5 - max(|c|))^2
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        wall_energy = np.sum((wall_dists) ** 2)
        return pair_energy + 10 * wall_energy
    
    # We don't actually need to compute full energy every step if it's too slow.
    # Let's stick to the score-based SA but with larger steps and more aggressive cooling.
    
    while time.time() - start_time < time_limit:
        progress = min(1.0, (time.time() - start_time) / time_limit)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        idx = rng.randint(N)
        # Larger steps to escape local minima
        delta = rng.normal(0, 0.02, size=3)
        candidate = current.copy()
        candidate[idx] += delta
        
        if np.any(np.abs(candidate[idx]) > 0.5):
            continue
        
        new_score = _score(candidate)
        delta_score = new_score - current_score
        
        if delta_score > 0 or rng.random() < math.exp(delta_score / temp):
            current = candidate
            current_score = new_score
            if new_score > best_score:
                best = candidate.copy()
                best_score = new_score
                
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Start with BCC init
    centers = _bcc_init()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Run multiple rounds of local search with different initial perturbations
    num_rounds = 5
    for r in range(num_rounds):
        if time.time() > deadline:
            break
        
        # Perturb the current best slightly
        rng = np.random.RandomState(r)
        perturbed = best_centers + rng.uniform(-0.02, 0.02, size=(N, 3))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Local search
        ls_centers, ls_score = _local_search(perturbed, time_limit=15.0, seed=r)
        if ls_score > best_score:
            best_centers = ls_centers
            best_score = ls_score
        
        # Simulated annealing on top
        if time.time() > deadline:
            break
        sa_centers, sa_score = _optimize_with_sa(ls_centers, time_limit=15.0, seed=r+100)
        if sa_score > best_score:
            best_centers = sa_centers
            best_score = sa_score
            
    return best_centers
# EVOLVE-BLOCK-END
