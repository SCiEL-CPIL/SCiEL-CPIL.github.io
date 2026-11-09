"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell using a structured approach:
    We use a combination of concentric rings and a central cross to create
    a complex dielectric pattern that maximizes the TM band gap between
    bands 4 and 5.
    
    The approach uses a deterministic pattern based on known photonic crystal
    designs that open large band gaps, combined with C4v symmetry enforcement.
    """
    size = problem["pixels"]
    
    # Create a base pattern using multiple harmonic components
    # This creates a complex, high-frequency texture
    cell = np.zeros((size, size), dtype=int)
    
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    for i in range(size):
        for j in range(size):
            dx = (i - cx) / size
            dy = (j - cy) / size
            r = np.sqrt(dx*dx + dy*dy)
            
            # Multiple ring structures at different radii
            # These create complex dielectric interfaces
            ring1 = np.cos(2 * np.pi * 3 * r) > 0.3
            ring2 = np.cos(2 * np.pi * 5 * r + np.pi/4) > 0.2
            ring3 = np.sin(2 * np.pi * 2 * r) > 0.1
            
            # Angular modulation for C4v symmetry
            theta = np.arctan2(dy, dx)
            ang1 = np.cos(4 * theta) > 0
            ang2 = np.sin(4 * theta) > 0
            
            # Combine features
            feature = (ring1 & ang1) | (ring2 & ang2) | (ring3 & (np.cos(2*theta) > 0))
            
            # Central region: solid dielectric
            if r < 0.08:
                cell[i, j] = 1
            # Outer boundary: air
            elif r > 0.48:
                cell[i, j] = 0
            else:
                cell[i, j] = 1 if feature else 0
    
    # Enforce C4v symmetry explicitly
    # 1. Horizontal flip symmetry
    cell = (cell + cell[:, ::-1]) // 2
    # 2. Vertical flip symmetry
    cell = (cell + cell[::-1, :]) // 2
    # 3. Transpose symmetry
    cell = (cell + cell.T) // 2
    # 4. Diagonal flip symmetry (anti-diagonal)
    cell = (cell + cell[::-1, ::-1]) // 2
    
    # Convert to nested list
    cell_list = cell.tolist()
    
    return cell_list
# EVOLVE-BLOCK-END
