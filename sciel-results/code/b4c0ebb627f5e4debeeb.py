"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # We will use a simple local search / optimization approach.
    # Start with a grid initialization and then refine using gradient-like updates
    # or a simple hill-climbing with random perturbations.
    
    # Better initialization: use a hexagonal or optimized grid
    # For 32 circles, a 6x6 grid gives 36 positions, use first 32
    # Or try a more compact arrangement.
    
    # Let's try a different approach: use scipy's minimize or a custom optimizer
    # Since we need to maximize sum of radii subject to non-overlap and boundary constraints,
    # this is a non-convex problem. We'll use a simple approach:
    # 1. Initialize with grid
    # 2. For each circle, try to increase radius until it hits another circle or boundary
    # 3. Iterate this process multiple times
    # 4. Also try random restarts with perturbed initial positions
    
    def grid_init(n, side=None):
        if side is None:
            side = int(np.ceil(np.sqrt(n)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < n:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    positions.append([x, y, 0.0])
        return np.array(positions[:n])
    
    def compute_radii(positions):
        """Compute max possible radii given fixed positions."""
        n = len(positions)
        radii = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            r_bound = min(positions[i][0], 1.0 - positions[i][0], 
                          positions[i][1], 1.0 - positions[i][1])
            # Distance to other circles
            r_min = r_bound
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                    r_min = min(r_min, dist / 2.0)
            radii[i] = max(0, r_min)
        return radii
    
    def total_radius(positions):
        return np.sum(compute_radii(positions))
    
    def refine(positions, max_iter=1000):
        """Iteratively adjust positions to increase total radius."""
        best_pos = positions.copy()
        best_score = total_radius(best_pos)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try moving each circle slightly in random directions
            for i in range(len(positions)):
                for _ in range(5):
                    dx = np.random.uniform(-0.02, 0.02)
                    dy = np.random.uniform(-0.02, 0.02)
                    new_pos = positions.copy()
                    new_pos[i][0] = np.clip(positions[i][0] + dx, 0, 1)
                    new_pos[i][1] = np.clip(positions[i][1] + dy, 0, 1)
                    score = total_radius(new_pos)
                    if score > best_score:
                        best_pos = new_pos.copy()
                        best_score = score
                        improved = True
            
            if not improved:
                break
        
        return best_pos, best_score
    
    # Try multiple initializations
    best_positions = None
    best_score = 0.0
    
    # Grid initialization
    pos = grid_init(N)
    pos, score = refine(pos, max_iter=500)
    if score > best_score:
        best_score = score
        best_positions = pos.copy()
    
    if time.time() > deadline:
        radii = compute_radii(best_positions)
        return np.column_stack([best_positions[:, :2], radii])
    
    # Random initialization
    for restart in range(3):
        if time.time() > deadline:
            break
        pos = np.random.uniform(0.1, 0.9, (N, 2))
        pos = np.column_stack([pos, np.zeros(N)])
        pos, score = refine(pos, max_iter=300)
        if score > best_score:
            best_score = score
            best_positions = pos.copy()
    
    # Final refinement with smaller steps
    if time.time() < deadline - 10:
        pos = best_positions.copy()
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(pos)):
                for _ in range(10):
                    dx = np.random.uniform(-0.005, 0.005)
                    dy = np.random.uniform(-0.005, 0.005)
                    new_pos = pos.copy()
                    new_pos[i][0] = np.clip(pos[i][0] + dx, 0, 1)
                    new_pos[i][1] = np.clip(pos[i][1] + dy, 0, 1)
                    score = total_radius(new_pos)
                    if score > best_score:
                        best_score = score
                        best_positions = new_pos.copy()
                        improved = True
            if not improved:
                break
    
    radii = compute_radii(best_positions)
    return np.column_stack([best_positions[:, :2], radii])
# EVOLVE-BLOCK-END
