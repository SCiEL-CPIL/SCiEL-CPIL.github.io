"""Circle packing in unit square n=32 using position-based dynamics."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 32 circles in the unit square [0,1]^2 maximizing sum of radii.
    Uses position-based dynamics: initialize with overlapping circles,
    iteratively expand radii and resolve penetrations via constraint projection.
    """
    deadline = time.time() + 140  # 140s safety margin within 150s limit
    
    # Initial positions: hexagonal packing pattern for better starting config
    # Use a jittered grid to break symmetry
    rng = np.random.default_rng(42)
    
    # Start with a reasonable grid layout
    side = int(np.ceil(np.sqrt(N)))  # 6x6 grid, use first 32
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            y = (j + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            positions.append([x, y])
    positions = np.array(positions[:N])
    
    # Initial small radius
    r0 = 0.02
    radii = np.full(N, r0)
    
    best_sum = 0.0
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    num_iter = 0
    max_iter = 50000
    
    # PBD parameters
    stiffness = 0.5  # How strongly we push apart
    expansion_rate = 0.0005  # How much to expand radii each iteration
    damping = 0.9
    
    while num_iter < max_iter and time.time() < deadline:
        # Step 1: Expand all radii slightly
        radii += expansion_rate
        # Cap radii so they don't exceed the square
        max_r = 0.5
        radii = np.minimum(radii, max_r)
        
        # Step 2: Resolve boundary constraints (project circles inside [0,1]^2)
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            # Clamp x
            if x - r < 0:
                positions[i, 0] = r
            elif x + r > 1:
                positions[i, 0] = 1 - r
            # Clamp y
            if y - r < 0:
                positions[i, 1] = r
            elif y + r > 1:
                positions[i, 1] = 1 - r
        
        # Step 3: Resolve pairwise penetration constraints
        for iter_pass in range(3):  # Multiple passes for better convergence
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        # Penetration: push apart
                        overlap = min_dist - dist
                        # Unit vector from i to j
                        if dist > 1e-12:
                            nx = dx / dist
                            ny = dy / dist
                        else:
                            # If exactly overlapping, use a random direction
                            angle = rng.uniform(0, 2 * np.pi)
                            nx = np.cos(angle)
                            ny = np.sin(angle)
                        
                        # Push each circle by half the overlap
                        push = 0.5 * overlap * stiffness
                        positions[i, 0] -= push * nx
                        positions[i, 1] -= push * ny
                        positions[j, 0] += push * nx
                        positions[j, 1] += push * ny
        
        # Step 4: Re-apply boundary constraints after pairwise resolution
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            positions[i, 0] = np.clip(x, r, 1 - r)
            positions[i, 1] = np.clip(y, r, 1 - r)
        
        # Check validity and update best
        valid = True
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                valid = False
                break
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        num_iter += 1
        
        # Adaptive: slow down expansion as we get closer to convergence
        if num_iter % 1000 == 0:
            expansion_rate *= 0.95
    
    # Final cleanup: ensure strict validity
    # Slightly shrink radii to ensure strict disjointness
    final_radii = best_radii * (1 - 1e-6)
    
    # Build result array
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = final_radii
    
    return result
# EVOLVE-BLOCK-END
