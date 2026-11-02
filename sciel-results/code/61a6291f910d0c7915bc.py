"""Improved circle packing: hexagonal lattice + gradient-based local search."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_lattice_init(n):
    """Generate a hexagonal lattice arrangement of n circles in a unit square."""
    # Use a hexagonal packing pattern
    # For n=21, try 5x5 grid with hex offset, or 4x6, etc.
    # Let's try several aspect ratios and pick the best
    
    best_positions = None
    best_radius = 0
    
    # Try different grid layouts
    for cols in range(3, 10):
        rows = int(np.ceil(n / cols))
        if cols * rows < n:
            continue
        
        # Hexagonal offset: alternate rows shifted by half
        # In a hex lattice, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
        # For a rectangle of width W and height H:
        # cols * 2r <= W  =>  r <= W/(2*cols)
        # (rows-1) * sqrt(3)*r + 2r <= H  =>  r <= H / (2 + (rows-1)*sqrt(3))
        
        # We want to maximize r subject to 2*(W+H) = 4, i.e., W+H = 2
        # Let's parameterize by aspect ratio
        for ar in np.linspace(0.4, 2.5, 50):
            W = ar
            H = 2 - W
            if H <= 0 or W <= 0:
                continue
            
            r1 = W / (2 * cols)
            r2 = H / (2 + (rows - 1) * np.sqrt(3))
            r = min(r1, r2)
            
            if r > best_radius:
                best_radius = r
                # Generate positions
                positions = []
                for j in range(rows):
                    for i in range(cols):
                        if len(positions) >= n:
                            break
                        x = (i + 0.5) * (2 * r) / W * W  # just i * 2r + r
                        # Actually let's place them properly
                        x = r + i * 2 * r
                        y = r + j * np.sqrt(3) * r
                        if j % 2 == 1:
                            x += r  # shift alternate rows
                        positions.append([x, y])
                    if len(positions) >= n:
                        break
                best_positions = np.array(positions[:n])
    
    if best_positions is None:
        # Fallback to square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Scale to fit in perimeter-4 rectangle
    xs = best_positions[:, 0]
    ys = best_positions[:, 1]
    W = xs.max() - xs.min() + 2 * best_radius
    H = ys.max() - ys.min() + 2 * best_radius
    
    # Scale so that 2*(W + H) = 4
    scale = 2.0 / (W + H)
    new_r = best_radius * scale
    new_positions = best_positions * scale
    # Center in rectangle
    new_positions[:, 0] -= xs.min() * scale
    new_positions[:, 1] -= ys.min() * scale
    
    return np.column_stack([new_positions, np.full(n, new_r)])


def _evaluate(positions):
    """Compute sum of radii given circle positions and radii."""
    return np.sum(positions[:, 2])


def _local_search(positions, max_iter=2000, step_size=0.01, deadline=None):
    """Simple gradient-free local search using random perturbations."""
    pos = positions.copy()
    n = pos.shape[0]
    best_val = _evaluate(pos)
    best_pos = pos.copy()
    
    start_time = time.time()
    
    for it in range(max_iter):
        if deadline is not None and time.time() - start_time > deadline:
            break
        
        # Try random perturbation
        trial = pos.copy()
        trial[np.random.random(n * 2) < 0.5, np.random.randint(0, 2, n)] += \
            np.random.uniform(-step_size, step_size, n)
        
        # Keep radii fixed, only move centers
        val = _evaluate(trial)
        
        # Check validity: all circles must be in a perimeter-4 rectangle
        # Find bounding box
        xs = trial[:, 0]
        ys = trial[:, 1]
        W = xs.max() - xs.min() + 2 * trial[:, 2].max()
        H = ys.max() - ys.min() + 2 * trial[:, 2].max()
        
        if 2 * (W + H) <= 4 + 1e-9:
            # Check no overlaps
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = trial[i, 0] - trial[j, 0]
                    dy = trial[i, 1] - trial[j, 1]
                    if dx * dx + dy * dy < (trial[i, 2] + trial[j, 2]) ** 2 - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if valid and val > best_val:
                best_val = val
                best_pos = trial.copy()
                pos = trial.copy()
    
    return best_pos


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Start with hexagonal lattice
    pos = _hex_lattice_init(N)
    
    # Local search
    pos = _local_search(pos, max_iter=5000, step_size=0.02, deadline=deadline - (time.time() - start_time))
    
    return pos
# EVOLVE-BLOCK-END
