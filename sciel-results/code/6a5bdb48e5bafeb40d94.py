import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin from 150s limit
    
    best_sum_r = 0
    best_positions = None
    best_radii = None
    
    # Helper: Check validity and compute score
    def check_and_score(x, y, r):
        # Boundary check
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False, 0
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False, 0
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False, 0
        
        return True, np.sum(r)

    # Helper: Resolve collisions (Projection)
    # This step ensures feasibility. We push overlapping circles apart.
    def resolve_collisions(x, y, r, iterations=100):
        for _ in range(iterations):
            moved = False
            for i in range(N):
                # Boundary push
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                elif x[i] > 1 - r[i]:
                    x[i] = 1 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                elif y[i] > 1 - r[i]:
                    y[i] = 1 - r[i]
                    moved = True
            
            # Pairwise push
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist < 1e-9:
                            # Random push if exactly overlapping
                            angle = random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        # Push apart by half the overlap each
                        overlap = min_dist - dist
                        # Simple equal mass assumption for stability, 
                        # though mass proportional to r^2 might be better.
                        # For now, equal push.
                        x[i] -= dx * overlap * 0.5
                        y[i] -= dy * overlap * 0.5
                        x[j] += dx * overlap * 0.5
                        y[j] += dy * overlap * 0.5
                        moved = True
            
            if not moved:
                break
                
        # Final boundary clamp (in case pushes went out of bounds)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        return x, y

    # Strategy: 
    # 1. Start with small radii, random positions.
    # 2. Iteratively increase radii.
    # 3. After each increase, resolve collisions.
    # 4. If resolution fails (circles stuck/overlapping significantly), 
    #    try to "shake" positions to escape local minima.
    # 5. Once max radius is reached, perform local search on positions 
    #    to maximize sum of radii (by allowing slight radius growth 
    #    and re-resolving).

    # Initial setup
    # A grid-like start is often better for packing
    side = int(np.ceil(np.sqrt(N)))
    
    # We will try a few different "growth" strategies
    # Strategy A: Uniform growth
    # Strategy B: Randomized growth
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        # Initialization
        x = np.random.uniform(0.2, 0.8, N)
        y = np.random.uniform(0.2, 0.8, N)
        
        # Start with very small radii to ensure feasibility
        r = np.full(N, 0.01)
        
        # Growth phase
        # We want to grow radii as much as possible.
        # The theoretical max sum of radii is bounded by area/geometry, 
        # but we don't know the exact bound. 
        # Let's grow in steps.
        
        current_r = 0.01
        max_possible_r = 0.5 # Theoretical max for a single circle in 1x1, but sum is constrained.
        
        # Adaptive step size
        step = 0.01
        
        while time.time() < deadline - 10:
            # Try to increase all radii by 'step'
            new_r = r + step
            
            # Check if any radius exceeds boundary constraints for its position
            # Actually, resolve_collisions handles boundaries, but if r is too big 
            # for the box (2*r > 1), it's impossible.
            if np.any(new_r > 0.5):
                break
                
            x_new, y_new = resolve_collisions(x.copy(), y.copy(), new_r, iterations=50)
            
            # Check if valid
            valid, score = check_and_score(x_new, y_new, new_r)
            
            if valid:
                x, y, r = x_new, y_new, new_r
                # Update best
                if score > best_sum_r:
                    best_sum_r = score
                    best_positions = x.copy()
                    best_radii = r.copy()
                    
                # If we succeeded, try a slightly larger step next time?
                # Or keep same step. Let's keep it stable.
                # If step was too big and we failed, we should reduce step.
                # Here we only update if valid, so step is fine.
                
                # Occasionally, try to "shake" to escape local minima 
                # and allow for larger growth
                if random.random() < 0.1:
                    # Small random perturbation
                    x_pert = x + np.random.normal(0, 0.01, N)
                    y_pert = y + np.random.normal(0, 0.01, N)
                    # Clamp to box
                    x_pert = np.clip(x_pert, r, 1-r)
                    y_pert = np.clip(y_pert, r, 1-r)
                    
                    x_res, y_res = resolve_collisions(x_pert, y_pert, r, iterations=50)
                    valid_p, _ = check_and_score(x_res, y_res, r)
                    if valid_p:
                        x, y = x_res, y_res
                        
            else:
                # Failed to grow. Reduce step size.
                step *= 0.5
                if step < 1e-4:
                    # Stuck. Try a larger perturbation to escape
                    # Or break and try a new restart
                    break
                    
        # Local Search Phase
        # Try to improve the sum of radii by fine-tuning positions
        # and allowing slight radius increases.
        if best_positions is not None and restart == 0: # Only do heavy local search on first restart?
             pass # Actually, let's do it for every restart if time permits
            
        if time.time() < deadline - 5:
            # Fine tuning
            # We have a feasible solution (x, y, r).
            # We want to maximize sum(r).
            # We can do a local search where we nudge positions 
            # and try to increase radii slightly.
            
            local_x = x.copy()
            local_y = y.copy()
            local_r = r.copy()
            
            local_step = 0.001
            for _ in range(1000):
                if time.time() > deadline - 2:
                    break
                    
                # Randomly pick a circle to move
                idx = random.randint(0, N-1)
                
                # Move it slightly
                angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(0, 0.02)
                new_x = local_x[idx] + dist * np.cos(angle)
                new_y = local_y[idx] + dist * np.sin(angle)
                
                # Clamp
                new_x = np.clip(new_x, local_r[idx], 1-local_r[idx])
                new_y = np.clip(new_y, local_r[idx], 1-local_r[idx])
                
                local_x[idx] = new_x
                local_y[idx] = new_y
                
                # Try to increase all radii slightly
                trial_r = local_r + 0.0005
                
                # Resolve collisions with new radii
                rx, ry = resolve_collisions(local_x.copy(), local_y.copy(), trial_r, iterations=20)
                
                valid, score = check_and_score(rx, ry, trial_r)
                
                if valid and score > best_sum_r:
                    best_sum_r = score
                    best_positions = rx.copy()
                    best_radii = trial_r.copy()
                    local_x = rx
                    local_y = ry
                    local_r = trial_r
                else:
                    # Revert position change if it didn't help
                    # (Simplified: just keep current local state, 
                    #  the position change might still be useful for future moves
                    #  even if radius didn't grow)
                    pass

    # Fallback if no solution found (should not happen with small start radii)
    if best_positions is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.01))
        best_positions = np.array([c[0] for c in cells])
        best_radii = np.array([c[2] for c in cells])
        best_sum_r = np.sum(best_radii)

    # Construct final output
    # Ensure strict disjointness by shrinking radii very slightly if needed
    # The check_and_score uses 1e-9 tolerance. 
    # To be safe against floating point issues in the evaluator, 
    # we can shrink radii by a tiny epsilon.
    
    # Let's verify the best solution one last time
    valid, final_score = check_and_score(best_positions, np.zeros(N), best_radii) # y is passed wrong here, fix
    valid, final_score = check_and_score(best_positions, np.zeros(N), best_radii) # Wait, y is not passed in check_and_score signature? 
    # Ah, check_and_score(x, y, r). I passed np.zeros(N) for y. That's wrong.
    
    # I need to store best_y as well.
    # Let's re-run the logic to ensure best_y is tracked.
    
    # Re-implementing the tracking properly in the main loop was implicit, 
    # but I didn't store best_y in the global scope correctly in the draft above.
    # Let's fix the code structure below.
    
    # For the sake of the final code block, I will rewrite the construct function 
    # to properly track best_y.
    
    return np.column_stack((best_positions, np.zeros(N), best_radii)) # Placeholder, will fix in final code

# Corrected Implementation

def construct():
    start_time = time.time()
    deadline = 145
    
    best_sum_r = 0
    best_x = None
    best_y = None
    best_r = None
    
    def check_and_score(x, y, r):
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False, 0
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False, 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False, 0
        return True, np.sum(r)

    def resolve_collisions(x, y, r, iterations=100):
        for _ in range(iterations):
            moved = False
            for i in range(N):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                elif x[i] > 1 - r[i]:
                    x[i] = 1 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                elif y[i] > 1 - r[i]:
                    y[i] = 1 - r[i]
                    moved = True
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist < 1e-9:
                            angle = random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        overlap = min_dist - dist
                        x[i] -= dx * overlap * 0.5
                        y[i] -= dy * overlap * 0.5
                        x[j] += dx * overlap * 0.5
                        y[j] += dy * overlap * 0.5
                        moved = True
            
            if not moved:
                break
                
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        return x, y

    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        x = np.random.uniform(0.2, 0.8, N)
        y = np.random.uniform(0.2, 0.8, N)
        r = np.full(N, 0.01)
        
        step = 0.01
        # Growth Phase
        while time.time() < deadline - 10:
            new_r = r + step
            if np.any(new_r > 0.5):
                break
                
            x_new, y_new = resolve_collisions(x.copy(), y.copy(), new_r, iterations=50)
            valid, score = check_and_score(x_new, y_new, new_r)
            
            if valid:
                x, y, r = x_new, y_new, new_r
                if score > best_sum_r:
                    best_sum_r = score
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
                    
                if random.random() < 0.1:
                    x_pert = x + np.random.normal(0, 0.01, N)
                    y_pert = y + np.random.normal(0, 0.01, N)
                    x_pert = np.clip(x_pert, r, 1-r)
                    y_pert = np.clip(y_pert, r, 1-r)
                    x_res, y_res = resolve_collisions(x_pert, y_pert, r, iterations=50)
                    valid_p, _ = check_and_score(x_res, y_res, r)
                    if valid_p:
                        x, y = x_res, y_res
            else:
                step *= 0.5
                if step < 1e-4:
                    break

        # Local Search Phase
        if time.time() < deadline - 5:
            local_x = x.copy()
            local_y = y.copy()
            local_r = r.copy()
            
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                idx = random.randint(0, N-1)
                angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(0, 0.02)
                new_x = local_x[idx] + dist * np.cos(angle)
                new_y = local_y[idx] + dist * np.sin(angle)
                new_x = np.clip(new_x, local_r[idx], 1-local_r[idx])
                new_y = np.clip(new_y, local_r[idx], 1-local_r[idx])
                
                local_x[idx] = new_x
                local_y[idx] = new_y
                
                trial_r = local_r + 0.0005
                rx, ry = resolve_collisions(local_x.copy(), local_y.copy(), trial_r, iterations=20)
                valid, score = check_and_score(rx, ry, trial_r)
                
                if valid and score > best_sum_r:
                    best_sum_r = score
                    best_x = rx.copy()
                    best_y = ry.copy()
                    best_r = trial_r.copy()
                    local_x = rx
                    local_y = ry
                    local_r = trial_r

    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.01))
        best_x = np.array([c[0] for c in cells])
        best_y = np.array([c[1] for c in cells])
        best_r = np.array([c[2] for c in cells])

    # Final safety shrink
    # The evaluator might be strict. Let's shrink radii by 1e-6 to be safe.
    # But we must ensure they don't overlap. Shrinking helps.
    final_r = best_r * (1 - 1e-6)
    
    # Verify one last time
    valid, _ = check_and_score(best_x, best_y, final_r)
    if not valid:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5/side - 0.01))
        best_x = np.array([c[0] for c in cells])
        best_y = np.array([c[1] for c in cells])
        best_r = np.array([c[2] for c in cells])
        final_r = best_r

    return np.column_stack((best_x, best_y, final_r))

# To satisfy the "complete program" requirement, we need to wrap this properly.
# The previous block had a duplicate construct definition. 
# I will output the clean version.

import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = 145
    
    best_sum_r = 0
    best_x = None
    best_y = None
    best_r = None
    
    def check_and_score(x, y, r):
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False, 0
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False, 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False, 0
        return True, np.sum(r)

    def resolve_collisions(x, y, r, iterations=100):
        for _ in range(iterations):
            moved = False
            for i in range(N):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                elif x[i] > 1 - r[i]:
                    x[i] = 1 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                elif y[i] > 1 - r[i]:
                    y[i] = 1 - r[i]
                    moved = True
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist < 1e-9:
                            angle = random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        overlap = min_dist - dist
                        x[i] -= dx * overlap * 0.5
                        y[i] -= dy * overlap * 0.5
                        x[j] += dx * overlap * 0.5
                        y[j] += dy * overlap * 0.5
                        moved = True
            
            if not moved:
                break
                
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        return x, y

    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        x = np.random.uniform(0.2, 0.8, N)
        y = np.random.uniform(0.2, 0.8, N)
        r = np.full(N, 0.01)
        
        step = 0.01
        while time.time() < deadline - 10:
            new_r = r + step
            if np.any(new_r > 0.5):
                break
                
            x_new, y_new = resolve_collisions(x.copy(), y.copy(), new_r, iterations=50)
            valid, score = check_and_score(x_new, y_new, new_r)
            
            if valid:
                x, y, r = x_new, y_new, new_r
                if score > best_sum_r:
                    best_sum_r = score
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
                    
                if random.random() < 0.1:
                    x_pert = x + np.random.normal(0, 0.01, N)
                    y_pert = y + np.random.normal(0, 0.01, N)
                    x_pert = np.clip(x_pert, r, 1-r)
                    y_pert = np.clip(y_pert, r, 1-r)
                    x_res, y_res = resolve_collisions(x_pert, y_pert, r, iterations=50)
                    valid_p, _ = check_and_score(x_res, y_res, r)
                    if valid_p:
                        x, y = x_res, y_res
            else:
                step *= 0.5
                if step < 1e-4:
                    break

        if time.time() < deadline - 5:
            local_x = x.copy()
            local_y = y.copy()
            local_r = r.copy()
            
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                idx = random.randint(0, N-1)
                angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(0, 0.02)
                new_x = local_x[idx] + dist * np.cos(angle)
                new_y = local_y[idx] + dist * np.sin(angle)
                new_x = np.clip(new_x, local_r[idx], 1-local_r[idx])
                new_y = np.clip(new_y, local_r[idx], 1-local_r[idx])
                
                local_x[idx] = new_x
                local_y[idx] = new_y
                
                trial_r = local_r + 0.0005
                rx, ry = resolve_collisions(local_x.copy(), local_y.copy(), trial_r, iterations=20)
                valid, score = check_and_score(rx, ry, trial_r)
                
                if valid and score > best_sum_r:
                    best_sum_r = score
                    best_x = rx.copy()
                    best_y = ry.copy()
                    best_r = trial_r.copy()
                    local_x = rx
                    local_y = ry
                    local_r = trial_r

    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.01))
        best_x = np.array([c[0] for c in cells])
        best_y = np.array([c[1] for c in cells])
        best_r = np.array([c[2] for c in cells])

    final_r = best_r * (1 - 1e-6)
    
    valid, _ = check_and_score(best_x, best_y, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5/side - 0.01))
        best_x = np.array([c[0] for c in cells])
        best_y = np.array([c[1] for c in cells])
        best_r = np.array([c[2] for c in cells])
        final_r = best_r

    return np.column_stack((best_x, best_y, final_r))
