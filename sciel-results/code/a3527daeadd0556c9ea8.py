import numpy as np
import random
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Vertices of a unit regular hexagon centered at origin, angle 0
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Normals to the sides of the outer hexagon (aligned with axes)
    # The sides are perpendicular to vectors at angles 30, 90, 150, 210, 270, 330
    outer_normals = np.array([
        [math.cos(math.pi/6 + i * math.pi / 3), math.sin(math.pi/6 + i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Precompute min and max projections of unit hexagon onto outer normals
    proj_matrix = hex_verts @ outer_normals.T  # 6x6
    max_proj_inner = np.max(proj_matrix, axis=0)  # (6,)
    min_proj_inner = np.min(proj_matrix, axis=0)  # (6,)
    
    # Helper to compute minimal enclosing hexagon side length for given positions
    def compute_S(pos):
        # pos is Nx2 array
        projs = pos @ outer_normals.T  # Nx6
        projs += max_proj_inner  # Nx6
        max_val = np.max(projs)
        S_req = max_val * 2.0 / math.sqrt(3)
        return S_req
    
    # Helper to check if two hexagons overlap and return penetration depth
    def check_overlap(pos_i, pos_j):
        # Use SAT with 6 axes
        dist_sq = np.sum((pos_i - pos_j)**2)
        if dist_sq > 4.0:  # Max distance for overlap is 2 (diagonal)
            return 0.0
        
        min_penetration = float('inf')
        
        for k in range(6):
            n = outer_normals[k]
            min_p_i = pos_i @ n + min_proj_inner[k]
            max_p_i = pos_i @ n + max_proj_inner[k]
            min_p_j = pos_j @ n + min_proj_inner[k]
            max_p_j = pos_j @ n + max_proj_inner[k]
            
            if max_p_i < min_p_j or max_p_j < min_p_i:
                return 0.0  # Separated
            
            pen = min(max_p_i, max_p_j) - max(min_p_i, min_p_j)
            if pen < min_penetration:
                min_penetration = pen
        
        return min_penetration
    
    # Helper to compute total overlap penalty
    def compute_penalty(pos):
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                pen = check_overlap(pos[i], pos[j])
                if pen > 0:
                    penalty += pen
        return penalty
    
    # Helper to compute total cost
    def compute_cost(pos):
        S = compute_S(pos)
        penalty = compute_penalty(pos)
        return S + penalty * 100.0  # High penalty to avoid overlaps

    # --- Discrete Grid Setup ---
    # Define a hexagonal grid with spacing epsilon
    eps = 0.1
    # Hex grid basis vectors
    # Pointy-top hex grid or flat-top? 
    # Let's use a standard axial coordinate system mapped to Cartesian.
    # q, r -> x = eps * (sqrt(3) * q + sqrt(3)/2 * r), y = eps * (3/2 * r)
    # This creates a grid of hexagons with "radius" eps (distance from center to vertex).
    # The distance between adjacent centers is sqrt(3) * eps.
    # For eps=0.1, dist = 0.173. This is fine-grained enough.
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = eps * (sqrt3 * q + 0.5 * sqrt3 * r)
        y = eps * (1.5 * r)
        return np.array([x, y])
    
    # Generate a set of candidate positions on the grid
    # We need a bounding box large enough to contain the packing.
    # Max S is likely around 3-4. So radius ~2.
    # Grid range: q, r from -20 to 20 (covers radius ~3.5)
    grid_range = 25
    candidates = []
    for q in range(-grid_range, grid_range + 1):
        for r in range(-grid_range, grid_range + 1):
            candidates.append(axial_to_cartesian(q, r))
    candidates = np.array(candidates)
    
    # To speed up search, we can precompute a mapping from position to index? 
    # Or just keep a list. Since N=12, and we move one at a time, we can just evaluate.
    
    # Directions for moves on the hex grid (6 neighbors)
    # (dq, dr)
    hex_moves = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # Map Cartesian position to grid index for quick lookup? 
    # It's easier to just store the (q, r) coordinates for each hexagon state.
    # State: list of (q, r) for each hexagon.
    
    def state_to_pos(state):
        return np.array([axial_to_cartesian(q, r) for q, r in state])
    
    # Initial state: generate a compact cluster
    def generate_initial_state():
        # Create a 4x3 block of hexes in axial coords
        # (0,0), (1,0), (2,0), (3,0)
        # (0,1), (1,1), (2,1), (3,1)
        # (0,2), (1,2), (2,2), (3,2)
        # This is 12 hexes.
        init_qr = []
        for i in range(4):
            for j in range(3):
                init_qr.append((i, j))
        return init_qr
    
    # Local Search on Discrete Grid
    def local_search(initial_qr, max_time=10.0):
        current_qr = [tuple(x) for x in initial_qr]
        current_pos = state_to_pos(current_qr)
        current_cost = compute_cost(current_pos)
        
        best_qr = [tuple(x) for x in current_qr]
        best_cost = current_cost
        best_pos = current_pos.copy()
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - start_time < max_time:
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random hexagon
            idx = random.randint(0, N - 1)
            old_qr = current_qr[idx]
            
            # Pick a random move direction
            dq, dr = random.choice(hex_moves)
            new_qr = (old_qr[0] + dq, old_qr[1] + dr)
            
            # Update state
            current_qr[idx] = new_qr
            # Only update the position of the moved hexagon? 
            # For simplicity, recompute full pos. N=12 is small.
            new_pos = state_to_pos(current_qr)
            new_cost = compute_cost(new_pos)
            
            if new_cost < current_cost:
                current_cost = new_cost
                current_pos = new_pos
                no_improve_count = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_qr = [tuple(x) for x in current_qr]
                    best_pos = new_pos.copy()
            else:
                # Revert
                current_qr[idx] = old_qr
                no_improve_count += 1
                
        return best_pos, best_cost

    # Multi-Start
    best_positions = None
    best_S = float('inf')
    
    num_starts = 10
    start_time_global = time.time()
    time_per_start = 10.0
    
    for start in range(num_starts):
        if time.time() - start_time_global > 110.0:
            break
            
        if start == 0:
            init_qr = generate_initial_state()
        else:
            # Perturb the initial state slightly
            init_qr = generate_initial_state()
            for i in range(N):
                if random.random() < 0.5:
                    dq, dr = random.choice(hex_moves)
                    init_qr[i] = (init_qr[i][0] + dq, init_qr[i][1] + dr)
        
        pos, cost = local_search(init_qr, max_time=time_per_start)
        
        # Verify no overlaps
        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if check_overlap(pos[i], pos[j]) > 0:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap:
            S = compute_S(pos)
            if S < best_S:
                best_S = S
                best_positions = pos.copy()
    
    # Fallback if no valid solution found (should not happen with high penalty)
    if best_positions is None:
        # Use the last one found or initial
        pos, cost = local_search(generate_initial_state(), max_time=10.0)
        best_positions = pos
        best_S = compute_S(pos)
        
    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
