import numpy as np
import random
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Vertices of a unit regular hexagon centered at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Normals to the sides of the outer hexagon (which is also aligned)
    # The sides are perpendicular to vectors at angles 30, 90, 150, 210, 270, 330
    # Wait, if outer hexagon has vertices at 0, 60..., its sides are perpendicular to 30, 90...
    # Yes.
    outer_normals = np.array([
        [math.cos(math.pi/6 + i * math.pi / 3), math.sin(math.pi/6 + i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Max projection of a unit hexagon onto these outer normals
    # This is a constant for all inner hexagons since they are aligned
    # proj = hex_verts @ outer_normals.T
    # max_proj_inner = np.max(proj, axis=0)
    max_proj_inner = np.max(hex_verts @ outer_normals.T, axis=0)
    
    # Initial positions: Honeycomb cluster
    # We want 12 hexagons.
    # A 4x3 block in a hex grid.
    # Let's generate coordinates for a hex grid of 12 cells.
    # i in 0..3, j in 0..2
    # x = i * 1.5 + j * 0.5
    # y = j * math.sqrt(3)/2
    # This creates a parallelogram.
    # Let's center it.
    
    positions = []
    # Generate 12 positions in a compact hex grid
    # Using axial coordinates (q, r)
    # We want a shape close to a hexagon itself.
    # A hexagon of side 2 has 1+6+12 = 19 cells. Too many.
    # A hexagon of side 1 has 7 cells.
    # We need 12.
    # Let's take a 3x4 rectangular patch of hex grid.
    for i in range(4):
        for j in range(3):
            x = i * 1.5 + j * 0.5
            y = j * math.sqrt(3) / 2
            positions.append([x, y])
            
    # Center the positions
    centers = np.array(positions)
    mean_c = np.mean(centers, axis=0)
    centers = centers - mean_c
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    max_iters = 50000 # Adjust based on time
    
    best_cost = float('inf')
    best_positions = centers.copy()
    
    def get_cost(pos):
        # pos is Nx2 array
        # 1. Calculate required S
        # For each hexagon i, and each normal k:
        # constraint: pos[i] @ n[k] + max_proj_inner[k] <= S * sqrt(3) / 2
        # S >= (pos[i] @ n[k] + max_proj_inner[k]) * 2 / sqrt(3)
        # We need max over all i, k
        
        # projections: pos (Nx2) @ outer_normals.T (2x6) -> Nx6
        projs = pos @ outer_normals.T
        # Add max_proj_inner (6,)
        projs += max_proj_inner
        # Max over i (axis 0) and k (axis 1)
        max_val = np.max(projs)
        S_req = max_val * 2.0 / math.sqrt(3)
        
        # 2. Calculate Overlap Penalty
        penalty = 0.0
        # Check all pairs
        for i in range(N):
            for j in range(i + 1, N):
                # Check if hex i and j overlap
                # Shift hex j by -pos[i]
                # We need to check if origin is inside Minkowski sum?
                # Or just SAT.
                # SAT: Check 6 axes.
                # If separated on any axis, no overlap.
                # If not separated on any, overlap.
                
                # To speed up, check distance first
                dist_sq = np.sum((pos[i] - pos[j])**2)
                # Min distance for non-overlap is sqrt(3) ~ 1.732
                # If dist > 2.0, definitely no overlap (diagonal is 2)
                if dist_sq > 4.0:
                    continue
                
                overlap = False
                min_penetration = float('inf')
                
                # Axes are the outer_normals? No, the normals to the sides of the hexagons.
                # Since they are aligned, the separating axes are the same 6 normals.
                for k in range(6):
                    n = outer_normals[k]
                    # Project pos[i] and pos[j] onto n
                    # Actually, we need to project the *vertices* of the hexagons.
                    # But since they are identical and aligned, the projection interval for a hexagon centered at c is:
                    # [c @ n - R, c @ n + R] where R = max projection of unit hex on n.
                    # Wait, the projection of the *set* of points is [min, max].
                    # For a regular hexagon, the projection onto any of its 6 normal axes is symmetric?
                    # The normal is perpendicular to a side. The side is at distance r_in = sqrt(3)/2.
                    # The opposite vertex is at distance 1.
                    # So the projection range is not symmetric around c @ n.
                    # Let's precompute min and max projections of unit hex on each normal.
                    
                    # We already computed max_proj_inner. We need min_proj_inner too.
                    # Let's compute it outside.
                    
                    min_p_i = pos[i] @ n + min_proj_inner[k]
                    max_p_i = pos[i] @ n + max_proj_inner[k]
                    
                    min_p_j = pos[j] @ n + min_proj_inner[k]
                    max_p_j = pos[j] @ n + max_proj_inner[k]
                    
                    # Check overlap on this axis
                    if max_p_i < min_p_j or max_p_j < min_p_i:
                        # Separated
                        overlap = False
                        break
                    else:
                        # Overlapping on this axis
                        # Penetration depth on this axis
                        pen = min(max_p_i, max_p_j) - max(min_p_i, min_p_j)
                        if pen < min_penetration:
                            min_penetration = pen
                        overlap = True
                
                if overlap:
                    penalty += min_penetration * 10.0 # Scaling factor
                    
        return S_req + penalty

    # Precompute min_proj_inner
    min_proj_inner = np.min(hex_verts @ outer_normals.T, axis=0)
    
    current_cost = get_cost(centers)
    
    for _ in range(max_iters):
        if T < min_T:
            break
            
        # Pick a random hexagon to move
        idx = random.randint(0, N - 1)
        old_pos = centers[idx].copy()
        
        # Move by random small vector
        # Step size should be related to T
        step_size = T * 0.5
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        
        centers[idx][0] += dx
        centers[idx][1] += dy
        
        new_cost = get_cost(centers)
        
        # Acceptance probability
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_positions = centers.copy()
        else:
            delta = new_cost - current_cost
            if delta < 0:
                # This shouldn't happen if new_cost < current_cost is checked above
                pass
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_cost = new_cost
                else:
                    # Reject
                    centers[idx] = old_pos
        
        T *= cooling

    # Final cleanup: try to minimize S further with local search on best_positions?
    # The SA should have found a good spot.
    
    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_cost), 0.0

# EVOLVE-BLOCK-END
