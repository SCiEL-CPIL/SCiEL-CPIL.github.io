"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Optimize D4-symmetric dielectric pattern for 4th-5th TM band gap using 2x2 rod array."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s budget
    
    # Coordinates centered at 0
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    y = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    def enforce_d4_symmetry(cell):
        """Average cell with its 8 D4 symmetry counterparts and threshold."""
        sym_cell = np.zeros_like(cell, dtype=float)
        # Identity
        sym_cell += cell
        # Rotations
        for k in range(1, 4):
            sym_cell += np.rot90(cell, k)
        # Reflections (Horizontal, Vertical, Diagonal, Anti-Diagonal)
        # Note: np.rot90(k) covers rotations. We need reflections too.
        # D4 group has 8 elements.
        # 0: Id
        # 1: Rot 90
        # 2: Rot 180
        # 3: Rot 270
        # 4: Flip Left-Right (Vertical axis)
        # 5: Flip Top-Bottom (Horizontal axis)
        # 6: Transpose (Diagonal)
        # 7: Anti-Transpose (Anti-Diagonal)
        
        sym_cell += np.fliplr(cell)       # Flip LR
        sym_cell += np.flipud(cell)       # Flip TB
        sym_cell += cell.T                # Transpose
        sym_cell += np.fliplr(cell.T)     # Anti-Transpose (or flipud(cell.T))
        
        # Threshold at 0.5 to get binary
        return (sym_cell > 0.5).astype(int)

    def make_2x2_rods(rod_size, offset):
        """
        Generate a 2x2 array of square rods.
        Centers at (offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset).
        rod_size is the half-width of the square rod.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Define the 4 centers
        centers = [
            (offset, offset),
            (offset, -offset),
            (-offset, offset),
            (-offset, -offset)
        ]
        
        for cx, cy in centers:
            # Mask for this rod
            mask = (np.abs(X - cx) <= rod_size) & (np.abs(Y - cy) <= rod_size)
            cell[mask] = 1
            
        # Enforce D4 symmetry to handle any discretization errors or overlap issues
        return enforce_d4_symmetry(cell)

    # Grid search over rod parameters
    # offset: distance from center to rod center. 
    # rod_size: half-width of the rod.
    # We want to explore different fill factors and spacings.
    
    best_cell = None
    best_score_proxy = -1.0
    
    # Search space
    # Offset from 0.1 to 0.4 (in units of cell width 1.0, so 0.1 is 10% from center)
    # Rod size from 0.05 to 0.2
    
    offsets = np.linspace(0.1, 0.4, 15)
    rod_sizes = np.linspace(0.05, 0.25, 15)
    
    for offset in offsets:
        for r_size in rod_sizes:
            if time.time() > deadline:
                break
                
            cell = make_2x2_rods(r_size, offset)
            
            # Heuristic scoring: 
            # 1. Check if it's not empty or full
            fill = cell.sum() / (size * size)
            if fill < 0.1 or fill > 0.9:
                continue
                
            # 2. Prefer specific fill factors known for photonic crystals?
            # Often ~30-50% is good.
            # Let's just pick the first valid one or optimize a simple proxy.
            # Since we can't run the band solver here, we rely on the structural knowledge:
            # "2x2 rod array enhances Bragg scattering".
            # We will select a configuration that is robust and symmetric.
            # Let's pick a mid-range configuration as a strong candidate.
            if best_cell is None:
                best_cell = cell
            
            # To "optimize" without the solver, we can look for patterns that have
            # strong Fourier components at the reciprocal lattice vectors.
            # However, simple grid search on geometry is safer.
            # Let's just iterate and keep the last valid one or one with specific properties.
            # Actually, let's just generate a few and return the most "standard" one.
            # A common good starting point for square lattice PC is rods at corners or center.
            # Here we have 4 rods.
            
            # Let's refine: We want to avoid rods touching each other or the boundary too much.
            # Max extent of a rod: offset + r_size.
            # If offset + r_size > 0.5, it touches boundary.
            if offset + r_size > 0.48:
                continue
            
            # Store this candidate. We'll just pick the one with fill factor closest to 0.4
            # as a heuristic for good interaction.
            score = -abs(fill - 0.4)
            if score > best_score_proxy:
                best_score_proxy = score
                best_cell = cell

    # Fallback if search failed
    if best_cell is None:
        # Default: 4 small rods
        best_cell = make_2x2_rods(0.1, 0.25)
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
