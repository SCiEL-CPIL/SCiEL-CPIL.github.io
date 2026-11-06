"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius for a given set of centers."""
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Minimum distance between pairs
    n = len(centers)
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    return min(min_pair_dist / 2.0, min_boundary)

def _hex_lattice_init(row_counts, dx, dy):
    """Create hex lattice centers with given row counts and spacings."""
    centers = []
    total_h = (len(row_counts) - 1) * dy
    y_offset = -total_h / 2.0
    
    for i, n_cols in enumerate(row_counts):
        y = y_offset + i * dy
        # Center this row horizontally
        if n_cols == 1:
            x_offset = 0.0
        else:
            # For hex packing, alternate rows are offset by dx/2
            row_width = (n_cols - 1) * dx
            x_offset = -row_width / 2.0
            # Add hex offset for odd rows
            if i % 2 == 1:
                x_offset += dx / 2.0
        for j in range(n_cols):
            x = x_offset + j * dx
            centers.append([x, y])
    
    return np.array(centers)

def _optimize_local(centers, start_time, deadline):
    """Local optimization using repulsion forces."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    alpha = 0.05
    step_scale = 1.0
    
    for it in range(2000):
        if time.time() - start_time > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        for i in range(N):
            # Boundary repulsion
            for k in range(2):
                if centers[i, k] > 0:
                    forces[i, k] += -alpha * max(0, centers[i, k])
                elif centers[i, k] < 0:
                    forces[i, k] += alpha * max(0, -centers[i, k])
            
            for j in range(N):
                if i == j:
                    continue
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                    continue
                # Repulsion force
                force_mag = alpha / (dist * dist)
                forces[i] += (d / dist) * force_mag
        
        # Update with adaptive step size
        centers += forces * step_scale
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            step_scale = min(step_scale * 1.05, 1.0)
        else:
            step_scale = max(step_scale * 0.9, 0.01)
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Try various hexagonal lattice configurations with alternating row lengths
    # Common patterns for 50 circles
    row_patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],  # 50
        [6, 7, 6, 7, 6, 7, 6, 5],  # 50
        [7, 7, 6, 7, 6, 7, 6, 4],  # 50
        [5, 7, 6, 7, 6, 7, 6, 6],  # 50
        [6, 6, 7, 6, 7, 6, 7, 5],  # 50
        [7, 6, 7, 6, 7, 6, 7, 4],  # 50 (duplicate for testing)
        [8, 6, 7, 6, 7, 6, 7, 3],  # 50
        [6, 7, 6, 7, 6, 7, 6, 5],  # 50
    ]
    
    # Try different spacings
    for dx in np.linspace(0.1, 0.2, 20):
        dy = dx * np.sqrt(3) / 2.0
        
        for pattern in row_patterns:
            if len(pattern) == 0 or sum(pattern) < N:
                continue
            
            centers = _hex_lattice_init(pattern, dx, dy)
            
            if len(centers) < N:
                continue
            
            # Check if all centers are within the square
            if np.any(np.abs(centers) > 0.5):
                continue
            
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
            
            if time.time() - start_time > deadline - 30:
                break
        
        if time.time() - start_time > deadline - 30:
            break
    
    # Also try uniform hex grids
    for n_cols in range(5, 12):
        for n_rows in range(5, 12):
            if n_cols * n_rows < N:
                continue
            
            dx = 2.0 / n_cols
            dy = dx * np.sqrt(3) / 2.0
            
            pattern = [n_cols] * n_rows
            centers = _hex_lattice_init(pattern, dx, dy)
            
            if len(centers) < N:
                continue
            
            if np.any(np.abs(centers) > 0.5):
                continue
            
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
            
            if time.time() - start_time > deadline - 30:
                break
        
        if time.time() - start_time > deadline - 30:
            break
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Local optimization
    best_centers, best_score = _optimize_local(best_centers, start_time, deadline)
    
    return best_centers
# EVOLVE-BLOCK-END
