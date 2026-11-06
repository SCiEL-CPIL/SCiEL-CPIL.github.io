"""Packing 50 equal circles in a unit square via hexagonal lattice + local optimization."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the maximum common radius for the given centers."""
    # Distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    # Min pairwise distance / 2
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(np.min(boundary), min_pair)


def _hexagonal_packing():
    """Generate initial hexagonal lattice packing with rows of alternating lengths."""
    # Try various row configurations that sum to 50
    # Hexagonal packing: rows offset by half spacing
    # Common configurations for 50:
    # 7+6+7+6+7+6+7+6+7+6 = 55 (too many)
    # 8+7+8+7+8+7+8+7 = 60 (too many)
    # 7+6+7+6+7+6+7+4 = 50
    # 6+7+6+7+6+7+6+5 = 50
    # 7+7+7+7+7+7+7+1 = 50 (bad)
    
    best_centers = None
    best_r = 0.0
    
    row_configs = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 7, 7, 7, 7, 7, 6, 2],
        [8, 7, 7, 7, 7, 6, 3, 5],
        [7, 6, 7, 7, 7, 6, 7, 3],
    ]
    
    for row_counts in row_configs:
        n_rows = len(row_counts)
        
        # Try various vertical spacings
        for vy in np.linspace(0.06, 0.12, 50):
            # Total height used
            total_h = (n_rows - 1) * vy
            
            # For hexagonal packing, horizontal spacing = 2*r, vertical offset = sqrt(3)/2 * 2*r
            # So vy = sqrt(3) * r => r = vy / sqrt(3)
            r = vy / np.sqrt(3)
            hx = 2 * r  # horizontal spacing
            
            # Vertical positions centered
            y_starts = (np.arange(n_rows) + 0.5) / n_rows - 0.5
            # Scale to fit
            y_pos = y_starts * (1.0 - 2 * r)
            
            # Check if fits
            if n_rows * vy > 1.0 - 2 * r:
                continue
            
            centers = []
            for i, count in enumerate(row_counts):
                y = y_pos[i]
                # Horizontal positions centered, with offset for hexagonal
                offset = (i % 2) * hx / 2.0
                x_starts = (np.arange(count) + 0.5) / count - 0.5
                x_pos = x_starts * (1.0 - 2 * r) + offset
                
                # Check if fits
                if count * hx > 1.0 - 2 * r:
                    continue
                
                for x in x_pos:
                    centers.append([x, y])
            
            if len(centers) != N:
                continue
            
            centers = np.array(centers)
            r_val = _score(centers)
            if r_val > best_r:
                best_r = r_val
                best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers


def _local_optimization(centers, max_iters=5000, lr=0.01):
    """Simple local optimization to improve the packing."""
    best_r = _score(centers)
    current = centers.copy()
    
    for it in range(max_iters):
        if it % 100 == 0 and time.time() > _deadline - 20:
            break
            
        # Compute gradients approximately
        # For each circle, compute forces pushing it apart from neighbors
        dists = cdist(current, current)
        np.fill_diagonal(dists, 1.0)
        
        # Boundary forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion (only for close pairs)
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.15:  # Only for close pairs
                    diff = current[i] - current[j]
                    force = 2.0 * (0.10 - d) / d if d > 1e-10 else 0.0
                    forces[i] += force * diff
                    forces[j] -= force * diff
        
        # Boundary attraction (push toward center if too close to edge)
        for i in range(N):
            for k in range(2):
                if current[i, k] > 0.45:
                    forces[i, k] -= 1.0
                elif current[i, k] < -0.45:
                    forces[i, k] += 1.0
        
        # Update
        current = current + lr * forces
        # Keep within bounds
        current = np.clip(current, -0.49, 0.49)
        
        r_val = _score(current)
        if r_val > best_r:
            best_r = r_val
            best_centers = current.copy()
        else:
            current = current - 0.5 * lr * forces  # Try smaller step
            r_val = _score(current)
            if r_val > best_r:
                best_r = r_val
                best_centers = current.copy()
    
    return best_centers, best_r


def construct():
    global _deadline
    _deadline = time.time() + 130  # Safety margin
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    best_r = _score(centers)
    best_centers = centers.copy()
    
    # Local optimization
    if time.time() < _deadline - 30:
        centers, r_val = _local_optimization(best_centers, max_iters=3000, lr=0.005)
        if r_val > best_r:
            best_r = r_val
            best_centers = centers.copy()
    
    # Second round of optimization with smaller steps
    if time.time() < _deadline - 15:
        centers, r_val = _local_optimization(best_centers, max_iters=2000, lr=0.002)
        if r_val > best_r:
            best_r = r_val
            best_centers = centers.copy()
    
    # Ensure validity
    assert best_centers.shape == (N, 2)
    assert np.all(np.isfinite(best_centers))
    assert np.all(np.abs(best_centers) <= 0.5 + 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
