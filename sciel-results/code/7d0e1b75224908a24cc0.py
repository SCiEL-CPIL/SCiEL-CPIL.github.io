import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # 1. Initialization
    # Use a grid to provide a decent starting point
    side = int(np.ceil(np.sqrt(N)))
    # For N=26, side=6. 6x6=36 cells. We use 26.
    # Arrange them in a 5x6 grid roughly?
    # 5 cols, 6 rows -> 30 cells.
    # Let's just use a grid of size ceil(sqrt(N)) and take the first N.
    # Or better, distribute them evenly.
    
    # Let's use a 5x6 grid for better distribution of 26 circles?
    # 5 * 6 = 30.
    # Indices: i in 0..4, j in 0..5.
    # Total 30. We need 26.
    # Let's stick to the simple grid from the seed but maybe adjust aspect ratio?
    # Seed used side=6 (sqrt(26)~5.1). 6x6 grid.
    # Let's use a 5x6 grid to be less dense initially?
    # Actually, a 6x6 grid with 26 circles leaves 10 empty spots.
    # Let's just use the grid coordinates.
    
    xs = []
    ys = []
    rs = []
    
    # 6x6 grid
    for i in range(6):
        for j in range(6):
            if len(xs) >= N:
                break
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            xs.append(x)
            ys.append(y)
            rs.append(0.05) # Start small
            
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    rs = np.array(rs[:N])
    
    # State: (N, 3) array [x, y, r]
    state = np.column_stack((xs, ys, rs))
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_energy(state):
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # Boundary Penalty
        # Violations: x < r, x > 1-r, y < r, y > 1-r
        vx = np.maximum(0, r - x) + np.maximum(0, x - (1 - r))
        vy = np.maximum(0, r - y) + np.maximum(0, y - (1 - r))
        p_boundary = np.sum(vx**2) + np.sum(vy**2)
        
        # Overlap Penalty
        # Distances between all pairs
        # Use broadcasting
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx**2 + dy**2)
        
        min_dist = r[i_idx] + r[j_idx]
        overlap = np.maximum(0, min_dist - d)
        p_overlap = np.sum(overlap**2)
        
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # Weights
        w_overlap = 100.0
        w_boundary = 100.0
        
        return obj + w_overlap * p_overlap + w_boundary * p_boundary

    # Simulated Annealing
    T_start = 0.05
    T_end = 0.0001
    # Number of iterations
    # Estimate cost per iteration.
    # Vectorized ops on N=26 are fast.
    # Let's try 20000 iterations.
    n_iters = 20000
    
    # Adaptive iterations based on time?
    # Let's just run a fixed number and check time.
    
    best_state = state.copy()
    best_energy = compute_energy(best_state)
    
    current_energy = best_energy
    
    # Cooling schedule
    # Geometric cooling
    alpha = (T_end / T_start) ** (1.0 / n_iters)
    
    for k in range(n_iters):
        if time.time() > deadline:
            break
            
        T = T_start * (alpha ** k)
        
        # Pick random index
        i = np.random.randint(N)
        
        # Propose move
        new_state = state.copy()
        
        # Perturb position
        dx = np.random.uniform(-0.02, 0.02)
        dy = np.random.uniform(-0.02, 0.02)
        new_state[i, 0] += dx
        new_state[i, 1] += dy
        
        # Perturb radius
        dr = np.random.uniform(-0.005, 0.005)
        new_state[i, 2] += dr
        
        # Clamp radius to positive
        new_state[i, 2] = max(0.001, new_state[i, 2])
        
        # Clamp position to [0,1] roughly? 
        # The penalty handles boundaries, but clamping helps convergence.
        # However, if we clamp, we might get stuck on the boundary.
        # Let's let the penalty handle it, but maybe clamp x,y to [-0.1, 1.1] to avoid huge penalties.
        new_state[i, 0] = np.clip(new_state[i, 0], -0.1, 1.1)
        new_state[i, 1] = np.clip(new_state[i, 1], -0.1, 1.1)
        
        new_energy = compute_energy(new_state)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            state = new_state
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = state.copy()
        
        # Optional: Restart if stuck?
        # Not implemented for simplicity.
        
    # Final Polish
    # Ensure validity.
    # If there are overlaps or boundary violations, we must fix them.
    # Strategy: Shrink radii until valid?
    # Or, run a local optimization to push apart.
    
    # Let's try a simple relaxation:
    # Iterate a few times:
    # 1. Check overlaps.
    # 2. If overlap, move centers apart.
    # 3. Check boundaries.
    # 4. If out of bounds, move in.
    # 5. If moving in causes overlap, shrink radius.
    
    # A safer final step:
    # Calculate the maximum allowed radius for each circle given current positions and other circles.
    # This is complex.
    
    # Simplest valid fallback:
    # Check if best_state is valid.
    # If not, shrink all radii by a small factor until valid.
    
    def is_valid(state):
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # Boundary
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
            
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-9:
                    return False
        return True
        
    # Try to polish best_state
    final_state = best_state.copy()
    
    # Check validity
    if not is_valid(final_state):
        # Shrink radii until valid
        # This is a worst-case fallback, might lose significant score
        scale = 1.0
        while not is_valid(final_state) and scale > 0.5:
            scale *= 0.95
            final_state[:, 2] *= scale
            
    # One more check: if still invalid, return a known valid grid
    if not is_valid(final_state):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        final_state = np.array(cells[:N])
        
    return final_state

# EVOLVE-BLOCK-START
# (The code above is the whole block)
# EVOLVE-BLOCK-END
