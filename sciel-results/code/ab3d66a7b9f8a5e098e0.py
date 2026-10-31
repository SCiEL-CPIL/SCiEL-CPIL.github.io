import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_score = 0.0
    best_config = None
    
    # Helper to check validity and score
    def evaluate(config):
        # config: (N, 3) array of x, y, r
        # Check bounds
        for i in range(N):
            x, y, r = config[i]
            if x < r - 1e-9 or y < r - 1e-9 or x > 1 - r + 1e-9 or y > 1 - r + 1e-9:
                return -1.0
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = config[i, 0] - config[j, 0]
                dy = config[i, 1] - config[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < config[i, 2] + config[j, 2] - 1e-9:
                    return -1.0
        return np.sum(config[:, 2])

    # Helper to resolve overlaps for circle i, returning new pos and validity flag
    def resolve_overlaps(i, pos, radii):
        x, y = pos
        r_i = radii[i]
        moved = False
        for _ in range(10): # Iterative resolution
            moved = False
            # Check boundaries
            if x < r_i:
                x = r_i
                moved = True
            elif x > 1 - r_i:
                x = 1 - r_i
                moved = True
            if y < r_i:
                y = r_i
                moved = True
            elif y > 1 - r_i:
                y = 1 - r_i
                moved = True
            
            # Check circles
            for j in range(N):
                if i == j: continue
                dx = x - radii[j][0] if False else x - 0 # placeholder, need positions
                # We need positions array
            # Let's pass positions array to this function or make it a method
            pass
        return x, y, moved

    # Better approach: Inline the logic in the main loop for speed and clarity
    
    def run_optimization(init_positions, init_radii):
        positions = init_positions.copy()
        radii = init_radii.copy()
        
        # Local optimization
        for step in range(5000):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            # Try to grow circle i
            old_r = radii[i]
            new_r = old_r + 0.005
            
            # Check if new_r is too big (max possible is 0.5, but practically limited)
            if new_r > 0.5:
                continue
                
            radii[i] = new_r
            
            # Resolve overlaps for i
            x, y = positions[i]
            valid = True
            max_iter = 50
            for _ in range(max_iter):
                moved = False
                # Bounds
                if x < new_r:
                    x = new_r
                    moved = True
                elif x > 1 - new_r:
                    x = 1 - new_r
                    moved = True
                if y < new_r:
                    y = new_r
                    moved = True
                elif y > 1 - new_r:
                    y = 1 - new_r
                    moved = True
                
                # Overlaps
                for j in range(N):
                    if i == j: continue
                    dx = x - positions[j, 0]
                    dy = y - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = new_r + radii[j]
                    if dist < min_dist:
                        if dist == 0:
                            # Random push
                            angle = np.random.uniform(0, 2*np.pi)
                            x = positions[j, 0] + np.cos(angle) * min_dist
                            y = positions[j, 1] + np.sin(angle) * min_dist
                        else:
                            # Push away
                            push_x = (min_dist - dist) * (dx / dist)
                            push_y = (min_dist - dist) * (dy / dist)
                            x += push_x
                            y += push_y
                        moved = True
                        # After pushing, we might violate bounds or other overlaps, so loop continues
                        # But we must ensure we don't push into the other circle again if we overshoot?
                        # The push is exact to boundary, so dist becomes min_dist.
                        # However, pushing i might push it into k. The loop handles that.
                
                if not moved:
                    break
            
            if moved:
                # Check if the final position is valid
                # The loop above tries to resolve, but might cycle or fail to converge
                # Simple check:
                final_valid = True
                if x < new_r - 1e-8 or x > 1 - new_r + 1e-8 or y < new_r - 1e-8 or y > 1 - new_r + 1e-8:
                    final_valid = False
                else:
                    for j in range(N):
                        if i == j: continue
                        dx = x - positions[j, 0]
                        dy = y - positions[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < new_r + radii[j] - 1e-8:
                            final_valid = False
                            break
                
                if final_valid:
                    positions[i] = [x, y]
                else:
                    # Revert
                    radii[i] = old_r
            else:
                # No move needed, radius increase accepted
                pass
        
        # Final evaluation
        score = evaluate(np.column_stack([positions, radii]))
        return score, np.column_stack([positions, radii])

    # Initializations
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_positions = []
    grid_radii = []
    for i in range(side):
        for j in range(side):
            if len(grid_positions) < N:
                grid_positions.append([(i + 0.5) / side, (j + 0.5) / side])
                grid_radii.append(0.01)
    
    # 2. Random
    random_positions = np.random.rand(N, 2) * 0.8 + 0.1
    random_radii = np.full(N, 0.01)

    init_strategies = [
        (np.array(grid_positions), np.array(grid_radii)),
        (random_positions, random_radii)
    ]
    
    # Add a few more random inits
    for _ in range(3):
        random_positions = np.random.rand(N, 2) * 0.8 + 0.1
        random_radii = np.full(N, 0.01)
        init_strategies.append((random_positions, random_radii))

    for init_pos, init_rad in init_strategies:
        if time.time() > deadline:
            break
            
        # Run optimization
        score, config = run_optimization(init_pos, init_rad)
        
        if score > best_score:
            best_score = score
            best_config = config
            
        # If we found a good solution, we might want to refine it further?
        # The loop inside run_optimization does refinement.
        
    # If no valid config found (shouldn't happen with small init radii), fallback to seed
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config

# EVOLVE-BLOCK-END
