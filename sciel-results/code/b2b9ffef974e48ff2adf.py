"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    
    def generate_patterns():
        """Generate a list of circle coordinates in unit square [0,1]x[0,1] for different topologies."""
        patterns = []
        
        # 1. 3x7 Grid
        coords = []
        for r in range(3):
            for c in range(7):
                coords.append(((c + 0.5) / 7.0, (r + 0.5) / 3.0))
        patterns.append(np.array(coords))
        
        # 2. 7x3 Grid
        coords = []
        for r in range(7):
            for c in range(3):
                coords.append(((c + 0.5) / 3.0, (r + 0.5) / 7.0))
        patterns.append(np.array(coords))
        
        # 3. 4x5 Grid + 1 Centered at Top
        coords = []
        for r in range(4):
            for c in range(5):
                coords.append(((c + 0.5) / 5.0, (r + 0.5) / 5.0))
        coords.append((0.5, 0.98))
        patterns.append(np.array(coords))
        
        # 4. 5x4 Grid + 1 Centered at Right
        coords = []
        for r in range(5):
            for c in range(4):
                coords.append(((c + 0.5) / 4.0, (r + 0.5) / 5.0))
        coords.append((0.98, 0.5))
        patterns.append(np.array(coords))
        
        # 5. 4x5 Grid + 1 Centered at Bottom
        coords = []
        for r in range(4):
            for c in range(5):
                coords.append(((c + 0.5) / 5.0, (r + 0.5) / 5.0))
        coords.append((0.5, 0.02))
        patterns.append(np.array(coords))
        
        # 6. 5x4 Grid + 1 Centered at Left
        coords = []
        for r in range(5):
            for c in range(4):
                coords.append(((c + 0.5) / 4.0, (r + 0.5) / 5.0))
        coords.append((0.02, 0.5))
        patterns.append(np.array(coords))
        
        return patterns

    patterns = generate_patterns()
    best_value = 0.0
    best_solution = None
    
    # Search over width W
    # Perimeter 4 => H = 2 - W
    # W in [0.5, 1.5]
    for W in np.linspace(0.5, 1.5, 2000):
        H = 2.0 - W
        
        for p_idx, coords in enumerate(patterns):
            # coords are in [0,1]x[0,1]
            # Scale to W x H
            xs = coords[:, 0] * W
            ys = coords[:, 1] * H
            
            # Calculate max radius based on boundaries
            r_bound_x = np.min(np.minimum(xs, W - xs))
            r_bound_y = np.min(np.minimum(ys, H - ys))
            r_bound = min(r_bound_x, r_bound_y)
            
            if r_bound <= 0:
                continue
                
            # Calculate max radius based on pairwise distances
            r_pair = np.inf
            n_circles = len(coords)
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_pair = min(r_pair, dist / 2.0)
            
            r = min(r_bound, r_pair)
            
            if r <= 0:
                continue
                
            value = N * r
            
            if value > best_value:
                best_value = value
                sol = np.zeros((N, 3))
                sol[:, 0] = xs
                sol[:, 1] = ys
                sol[:, 2] = r
                best_solution = sol
                
    if best_solution is None:
        # Fallback: Simple 3x7 grid in 1x1
        coords = []
        for r in range(3):
            for c in range(7):
                coords.append(((c + 0.5) / 7.0, (r + 0.5) / 3.0))
        coords = np.array(coords)
        W = 1.0
        H = 1.0
        xs = coords[:, 0] * W
        ys = coords[:, 1] * H
        r_bound_x = np.min(np.minimum(xs, W - xs))
        r_bound_y = np.min(np.minimum(ys, H - ys))
        r_bound = min(r_bound_x, r_bound_y)
        r_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_pair = min(r_pair, dist / 2.0)
        r = min(r_bound, r_pair)
        sol = np.zeros((N, 3))
        sol[:, 0] = xs
        sol[:, 1] = ys
        sol[:, 2] = r
        best_solution = sol

    return best_solution
# EVOLVE-BLOCK-END
