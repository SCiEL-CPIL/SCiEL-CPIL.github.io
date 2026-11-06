"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """Concentric square rings with alternating dielectric/air for strong TM band gap."""
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    # Center is at (15.5, 15.5) for 32x32
    center = (size - 1) / 2.0  # 15.5
    
    # Assign each pixel to a ring based on Chebyshev distance
    for i in range(size):
        for j in range(size):
            dist = max(abs(i - center), abs(j - center))
            ring = int(dist - 0.5)  # 0 to 15
            # Alternate: even rings dielectric, odd rings air
            cell[i][j] = 1 if ring % 2 == 0 else 0
    
    return cell
# EVOLVE-BLOCK-END
