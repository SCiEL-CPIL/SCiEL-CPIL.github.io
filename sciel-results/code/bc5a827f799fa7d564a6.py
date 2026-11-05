import time

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Define the 5 points to remove from the R=3 hexagon to achieve 76 edges
    # The R=3 hexagon has 37 points and 90 edges.
    # Removing 4 corners (deg 3) and 1 side point adjacent to one corner (deg 4)
    # Results in lost edges = 3*4 + 4 - 2*1 = 16 - 2 = 14.
    # Remaining edges = 90 - 14 = 76.
    remove_set = {(3, 0), (0, 3), (-3, 3), (-3, 0), (2, 1)}
    
    points = []
    R = 3
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R:
                if (a, b) not in remove_set:
                    points.append([a, b])
                    
    # Verify count
    if len(points) != 32:
        # Fallback: just return 32 points from the grid, though this shouldn't happen
        return [[a, b] for a in range(16) for b in range(2)]
        
    return points
