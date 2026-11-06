"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate the first 32 points of a hexagonal spiral.
    # This shape is known to maximize the number of edges for a given number of points
    # on a triangular lattice (Harborth's theorem).
    
    # Directions in axial coordinates for the spiral steps
    # The spiral moves in 6 directions, with step lengths increasing every 2 steps.
    # Dir 0: (1, 0)   (East)
    # Dir 1: (1, -1)  (South-East)
    # Dir 2: (0, -1)  (South-West)
    # Dir 3: (-1, 0)  (West)
    # Dir 4: (-1, 1)  (North-West)
    # Dir 5: (0, 1)   (North-East)
    
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    points = [(0, 0)]
    
    if len(points) >= 32:
        return [list(p) for p in points]
        
    current = (0, 0)
    step = 0
    
    while len(points) < 32:
        # Determine direction and step length
        dir_idx = step % 6
        step_len = (step // 2) + 1
        
        dq, dr = dirs[dir_idx]
        
        for _ in range(step_len):
            if len(points) >= 32:
                break
            current = (current[0] + dq, current[1] + dr)
            points.append(current)
            
        step += 1
        
    # Convert to list of lists
    result = [list(p) for p in points]
    return result
# EVOLVE-BLOCK-END
