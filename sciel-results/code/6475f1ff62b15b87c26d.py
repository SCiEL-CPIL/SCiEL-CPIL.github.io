"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def design(problem):
    """
    Inverted photonic crystal: dielectric matrix (eps=12) with hexagonal air holes.
    The hexagonal lattice of air holes in a dielectric background is the canonical
    structure for large TM bandgaps in square-lattice PCs.
    
    Design: 32x32 cell, dielectric=1 everywhere, air holes (0) at hexagonal positions.
    Holes are circular with radius ~0.18-0.22 lattice constants, arranged in a
    hexagonal pattern that respects D4 symmetry of the square unit cell.
    """
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 230s kill
    
    # Start with all dielectric (1)
    cell = [[1] * size for _ in range(size)]
    
    # Hexagonal air-hole lattice in dielectric background
    # For a 32x32 cell, we place holes in a hexagonal pattern.
    # The square unit cell of a hexagonal lattice has lattice constant a = 32 pixels.
    # Holes at fractional positions that create D4 symmetry.
    
    # Strategy: place air holes at positions that form a hexagonal-like pattern
    # within the square cell, maintaining 180-degree rotational and reflection symmetry.
    
    # Hexagonal lattice points in a square cell (fractional coords 0..1):
    # For a hexagonal lattice with lattice constant a, in a square supercell of side a:
    # Primary hole at center: (0.5, 0.5)
    # Secondary holes at hexagonal neighbors projected into the cell
    # Hexagonal neighbors at 60-degree angles: distance a/sqrt(3) from center
    # In fractional coords: (0.5 + 1/sqrt(3), 0.5), etc. - but these wrap around.
    
    # For D4 symmetry in a square cell, use a pattern with holes at:
    # Center and 4 symmetric positions forming a cross/diamond pattern
    # This approximates hexagonal packing while maintaining square symmetry.
    
    # Hole radius (fraction of cell size)
    hole_radius = 0.18
    
    # Hole centers (in fractional coordinates, 0 to 1)
    # D4-symmetric arrangement approximating hexagonal:
    # Center hole + 4 holes at (0.5±r, 0.5) and (0.5, 0.5±r) where r creates hex-like spacing
    # For hexagonal, neighbor distance = a/sqrt(3) ≈ 0.577a
    # So offset from center ≈ 0.288 in fractional coords
    
    hex_offset = 0.2887  # 1/(2*sqrt(3)) ≈ 0.2887
    
    hole_centers = [
        (0.5, 0.5),           # Center
        (0.5 + hex_offset, 0.5),  # Right
        (0.5 - hex_offset, 0.5),  # Left
        (0.5, 0.5 + hex_offset),  # Top
        (0.5, 0.5 - hex_offset),  # Bottom
    ]
    
    # Also add corner-proximate holes for denser packing (hexagonal has 6 neighbors)
    # At 30-degree and 90-degree from horizontal:
    # (0.5 + hex_offset*cos(30), 0.5 + hex_offset*sin(30))
    hex_offset_diag = hex_offset
    diag_x = hex_offset_diag * math.cos(math.pi/6)  # ~0.25
    diag_y = hex_offset_diag * math.sin(math.pi/6)  # ~0.144
    
    # Add diagonal holes (4 more for full hexagonal 6-neighbor + center)
    # But for D4 symmetry, add all 4 diagonals
    hole_centers.extend([
        (0.5 + diag_x, 0.5 + diag_y),
        (0.5 - diag_x, 0.5 + diag_y),
        (0.5 + diag_x, 0.5 - diag_y),
        (0.5 - diag_x, 0.5 - diag_y),
    ])
    
    # Carve out circular air holes
    for cx_frac, cy_frac in hole_centers:
        cx_px = cx_frac * size
        cy_px = cy_frac * size
        r_px = hole_radius * size
        
        # Determine pixel range
        i_min = max(0, int(math.floor(cy_px - r_px)))
        i_max = min(size - 1, int(math.ceil(cy_px + r_px)))
        j_min = max(0, int(math.floor(cx_px - r_px)))
        j_max = min(size - 1, int(math.ceil(cx_px + r_px)))
        
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # Pixel center
                px = (j + 0.5) - cx_px
                py = (i + 0.5) - cy_px
                if px * px + py * py <= r_px * r_px:
                    cell[i][j] = 0
    
    # Verify D4 symmetry and fix if needed
    # Check: cell[i][j] == cell[size-1-i][j] (vertical reflection)
    #         cell[i][j] == cell[i][size-1-j] (horizontal reflection)
    #         cell[i][j] == cell[size-1-j][size-1-i] (90-degree rotation)
    
    # Enforce symmetry by averaging (rounding)
    for i in range(size):
        for j in range(size):
            # Collect symmetric equivalents
            vals = [
                cell[i][j],
                cell[size-1-i][j],       # vertical flip
                cell[i][size-1-j],       # horizontal flip
                cell[size-1-i][size-1-j], # 180 rot
                cell[j][i],              # transpose
                cell[size-1-j][size-1-i], # transpose + 180
                cell[j][size-1-i],       # 90 rot CW
                cell[size-1-j][i],       # 90 rot CCW
            ]
            avg = sum(vals) / len(vals)
            cell[i][j] = 1 if avg >= 0.5 else 0
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return cell
    
    # Try a few radius variations if time permits
    best_cell = cell
    best_score_estimate = -1  # We can't compute the actual band gap, so use heuristic
    
    # Heuristic: larger holes in dielectric background generally help
    # Try slightly different radii
    for radius_frac in [0.17, 0.19, 0.20, 0.21]:
        if time.time() - start_time > deadline - 10:
            break
        
        test_cell = [[1] * size for _ in range(size)]
        
        for cx_frac, cy_frac in hole_centers:
            cx_px = cx_frac * size
            cy_px = cy_frac * size
            r_px = radius_frac * size
            
            i_min = max(0, int(math.floor(cy_px - r_px)))
            i_max = min(size - 1, int(math.ceil(cy_px + r_px)))
            j_min = max(0, int(math.floor(cx_px - r_px)))
            j_max = min(size - 1, int(math.ceil(cx_px + r_px)))
            
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    px = (j + 0.5) - cx_px
                    py = (i + 0.5) - cy_px
                    if px * px + py * py <= r_px * r_px:
                        test_cell[i][j] = 0
        
        # Enforce D4 symmetry
        for i in range(size):
            for j in range(size):
                vals = [
                    test_cell[i][j],
                    test_cell[size-1-i][j],
                    test_cell[i][size-1-j],
                    test_cell[size-1-i][size-1-j],
                    test_cell[j][i],
                    test_cell[size-1-j][size-1-i],
                    test_cell[j][size-1-i],
                    test_cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                test_cell[i][j] = 1 if avg >= 0.5 else 0
        
        # Heuristic score: prefer structures with moderate fill fraction
        # In inverted PC, dielectric fraction should be ~0.5-0.7 for good gaps
        ones = sum(sum(row) for row in test_cell)
        frac = ones / (size * size)
        # Optimal dielectric fraction for TM gap is around 0.55-0.65
        if 0.50 <= frac <= 0.70:
            # Prefer closer to 0.60
            score_est = 1.0 - abs(frac - 0.60)
            if score_est > best_score_estimate:
                best_score_estimate = score_est
                best_cell = test_cell
    
    return best_cell
# EVOLVE-BLOCK-END
