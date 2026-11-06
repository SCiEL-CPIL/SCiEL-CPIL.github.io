"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit
    
    # Generate candidate cells
    # We focus on high-symmetry shapes: Circular and Square holes/rods.
    # The gap between band 4 and 5 in a square lattice PC is typically maximized 
    # by a structure with a moderate fill fraction (~0.5-0.6) and high symmetry.
    
    candidates = []
    
    # Helper to create a cell
    def make_cell(is_hole, shape_type, radius):
        """
        is_hole: True if the central feature is air (0) in dielectric (1)
                 False if the central feature is dielectric (1) in air (0)
        shape_type: 'circle' or 'square'
        radius: half-width of the central feature
        """
        cell = [[0] * size for _ in range(size)]
        center = size / 2.0
        val_feature = 0 if is_hole else 1
        val_bg = 1 if is_hole else 0
        
        for i in range(size):
            for j in range(size):
                # Coordinates relative to center, 0.5 offset for pixel centers
                x = i + 0.5 - center
                y = j + 0.5 - center
                
                if shape_type == 'circle':
                    dist_sq = x*x + y*y
                    if dist_sq < radius*radius:
                        cell[i][j] = val_feature
                    else:
                        cell[i][j] = val_bg
                elif shape_type == 'square':
                    # Aligned square
                    if abs(x) < radius and abs(y) < radius:
                        cell[i][j] = val_feature
                    else:
                        cell[i][j] = val_bg
                elif shape_type == 'diamond':
                    # Rotated 45 degrees
                    if abs(x) + abs(y) < radius * 1.414: # approx radius in diamond
                        cell[i][j] = val_feature
                    else:
                        cell[i][j] = val_bg
                        
        return cell

    # Radii to try (in pixels, half-width)
    # Cell is 32x32, so max radius is 16.
    # Fill fraction approx 4*R^2 / 32^2 for square, pi*R^2/32^2 for circle.
    # Target fill fraction ~0.5.
    # For square: 4*R^2/1024 = 0.5 -> R^2 = 128 -> R ~ 11.3
    # For circle: pi*R^2/1024 = 0.5 -> R^2 = 162 -> R ~ 12.7
    
    radii = [8, 9, 10, 11, 12]
    shapes = ['circle', 'square', 'diamond']
    polarities = [True, False] # True=Hole, False=Rod
    
    for pol in polarities:
        for shape in shapes:
            for r in radii:
                if r > 15: continue
                cell = make_cell(pol, shape, r)
                # Verify symmetry (should be guaranteed by construction, but safe to check)
                if _check_symmetry(cell, size):
                    candidates.append(cell)
    
    # Heuristic Selection:
    # In square lattice PCs, the gap between bands 4 and 5 is often the gap at the X-point 
    # for the "hole" case (if the first gap is at Gamma/X). 
    # Actually, for TM, the first gap is often at X. The gap between 4 and 5 might be at M or X.
    # Circular holes are generally robust. 
    # Let's pick a Circular Hole with R=10 (Fill ~0.38) or R=11 (Fill ~0.47).
    # Or a Square Hole with R=10 (Fill ~0.39) or R=11 (Fill ~0.48).
    
    # Based on typical PC literature, a fill fraction of ~0.5 is good.
    # Let's select the Circular Hole with R=11.
    best_cell = None
    for cell in candidates:
        # Simple heuristic: Count ones. 
        # For holes in dielectric, we want ~50% air.
        # For rods in air, we want ~50% dielectric.
        ones = sum(sum(row) for row in cell)
        fill = ones / (size*size)
        
        # We prioritize circular holes (isotropic, high symmetry)
        # Identify which cell this is? We lost the metadata. 
        # Let's just regenerate the specific "best guess" cell directly.
        
    # Directly generate the best guess: Circular Hole, R=11
    # Why R=11? 
    # R=11 -> Area ~ 380 px. Total 1024. Air fraction ~ 0.37. Dielectric ~ 0.63.
    # R=12 -> Area ~ 452 px. Air ~ 0.44. Dielectric ~ 0.56.
    # R=10 -> Area ~ 314 px. Air ~ 0.30. Dielectric ~ 0.70.
    
    # Let's try R=11 for the hole.
    best_cell = make_cell(True, 'circle', 11)
    
    # Alternative: Square Hole R=10
    # cell_sq = make_cell(True, 'square', 10)
    
    # If time permits, we could implement a local search, but we don't have the solver.
    # So we return the best heuristic guess.
    
    return best_cell

def _check_symmetry(cell, size):
    """Check if cell has square symmetry: 180-degree rotation, transpose, and reflections."""
    # Check 180-degree rotation symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                return False
    
    # Check transpose symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                return False
    
    # Check horizontal reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                return False
    
    # Check vertical reflection
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                return False
    
    return True
# EVOLVE-BLOCK-END
