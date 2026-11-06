import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Directions for the hexagonal spiral in axial coordinates (a, b)
    # The lattice is defined by z = a + b*omega, omega = exp(i*pi/3)
    # Neighbors of (0,0) are:
    # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    # Let's verify distances:
    # d^2(a1-b1, a2-b2) = da^2 + da*db + db^2
    # (1,0): 1
    # (0,1): 1
    # (-1,1): 1 -1 + 1 = 1
    # (-1,0): 1
    # (0,-1): 1
    # (1,-1): 1 -1 + 1 = 1
    # These are the 6 neighbors.
    
    dirs = [
        (1, 0),   # East
        (0, 1),   # North-East
        (-1, 1),  # North-West
        (-1, 0),  # West
        (0, -1),  # South-West
        (1, -1)   # South-East
    ]
    
    points = [(0, 0)]
    
    if len(points) >= 32:
        return [[a, b] for a, b in points]
        
    step_size = 1
    dir_index = 0
    
    while len(points) < 32:
        # In the hexagonal spiral, the step size increases every 2 sides
        # Side 0: 1 step
        # Side 1: 1 step
        # Side 2: 2 steps
        # Side 3: 2 steps
        # Side 4: 3 steps
        # Side 5: 3 steps
        # Side 6: 4 steps
        # ...
        # So for side k, the number of steps is floor(k/2) + 1?
        # k=0 -> 1
        # k=1 -> 1
        # k=2 -> 2
        # k=3 -> 2
        # k=4 -> 3
        # k=5 -> 3
        # Yes, steps = dir_index // 2 + 1
        
        steps_needed = dir_index // 2 + 1
        
        da, db = dirs[dir_index % 6]
        
        for _ in range(steps_needed):
            # Move one step in the current direction
            last_a, last_b = points[-1]
            new_a = last_a + da
            new_b = last_b + db
            
            # Check bounds
            if abs(new_a) <= 64 and abs(new_b) <= 64:
                points.append((new_a, new_b))
                
                if len(points) >= 32:
                    break
        else:
            # If we completed the steps and didn't reach 32, move to next direction
            dir_index += 1
            continue
            
        if len(points) >= 32:
            break
            
        # If we broke out of the inner loop because we reached 32, we are done.
        # Otherwise, we completed the steps for this direction and should advance.
        # The logic above with 'else' on 'for' handles the normal completion.
        # If we broke, we exit the while loop.
        
    # Ensure we return exactly 32 points
    # If for some reason we have more (shouldn't happen with break), slice
    points = points[:32]
    
    # Convert to list of lists
    return [[a, b] for a, b in points]
