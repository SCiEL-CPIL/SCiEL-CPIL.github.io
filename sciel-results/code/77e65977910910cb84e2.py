"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Hardcoded optimal 32-point subset of the R=3 hexagon
    # Achieves 76 edges (Harborth's optimum for n=32)
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a + b) <= 3:
                points.append([a, b])
    
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    result = [p for p in points if (p[0], p[1]) not in remove_set]
    return result
# EVOLVE-BLOCK-END
