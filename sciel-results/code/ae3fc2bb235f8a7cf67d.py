"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def design(problem):
    """Design a 32x32 photonic-crystal unit cell maximizing the TM 4-5 band gap."""
    size = problem["pixels"]
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s budget
    
    # Pre-compute symmetry: we only need to decide pixels in the fundamental domain
    # For a 32x32 cell with 4-fold symmetry about center:
    # The cell is symmetric under: row reversal, column reversal, and transpose
    # Fundamental domain: top-left quadrant (rows 0-15, cols 0-15)
    # Due to symmetry, pixel (i,j) determines:
    #   (i,j), (i, 31-j), (31-i, j), (31-i, 31-j)
    # Also transpose symmetry means (i,j) == (j,i)
    
    # For 4-fold + mirror symmetry, the fundamental domain is the upper-left triangle
    # including the diagonal, of the upper-left quadrant
    # Let's work with the upper-left 16x16 quadrant, with the constraint that
    # it's symmetric about the main diagonal (i<=j for the fundamental region)
    
    # Actually, let me think more carefully:
    # cell[i][j] == cell[i][31-j]  (horizontal mirror)
    # cell[i][j] == cell[31-i][j]  (vertical mirror)  
    # cell[i][j] == cell[j][i]     (transpose)
    
    # The fundamental domain is: i in [0,15], j in [0,15], j >= i (upper triangle of TL quadrant)
    # That gives 16*17/2 = 136 independent pixels
    
    # But we can do better by using a smarter parameterization.
    # Let's try concentric square rings approach with some randomization.
    
    # Strategy: Try several analytical shapes and also do local optimization
    
    def make_ring_pattern(ring_assignments, size=32):
        """Create a cell from concentric square rings.
        ring_assignments: list of 0/1, index = ring number from center outward
        For 32x32, there are 16 rings (distance from center in Chebyshev metric)
        Ring 0 = center 2x2, ring 1 = next shell, etc.
        Actually for even size, center is between pixels.
        Let's define ring by Chebyshev distance from center (15.5, 15.5)
        """
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Chebyshev distance from center (15.5, 15.5)
                dist = max(abs(i - 15.5), abs(j - 15.5))
                ring = int(dist)  # 0 to 15
                cell[i][j] = ring_assignments[ring]
        return cell
    
    def make_circle_pattern(radius, size=32):
        """Create a circular dielectric region."""
        cell = [[0] * size for _ in range(size)]
        cx = cy = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy <= radius*radius:
                    cell[i][j] = 1
        return cell
    
    def make_superellipse(a, n_exp, size=32):
        """Super-ellipse rod."""
        cell = [[0] * size for _ in range(size)]
        cx = cy = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs(i - cx) / a
                dy = abs(j - cy) / a
                if dx**n_exp + dy**n_exp < 1.0:
                    cell[i][j] = 1
        return cell
    
    def is_valid(cell, size=32):
        """Check 4-fold symmetry."""
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[i][size-1-j]:
                    return False
                if cell[i][j] != cell[size-1-i][j]:
                    return False
                if cell[i][j] != cell[j][i]:
                    return False
        return True
    
    def quick_gap_estimate(cell, size=32):
        """Very rough estimate of dielectric fraction for ranking candidates."""
        dielectric = sum(sum(row) for row in cell)
        # Return dielectric fraction as a proxy (not actual gap)
        return dielectric / (size * size)
    
    # Generate candidate shapes
    candidates = []
    
    # 1. Various circle radii
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        candidates.append(make_circle_pattern(r))
    
    # 2. Super-ellipses
    for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for n in [2, 4, 8, 16]:
            candidates.append(make_superellipse(a, n))
    
    # 3. Ring patterns (alternating)
    # Try various ring assignment patterns
    import random
    random.seed(42)
    
    # Start with the seed pattern and try perturbations
    base_cell = make_circle_pattern(0.2)
    candidates.append(base_cell)
    
    # Try flipping individual pixels in the fundamental domain
    # Fundamental domain: upper-left triangle of 16x16
    # Let's try random flips
    
    best_candidate = base_cell
    best_score_proxy = 0.5  # Start with 50% dielectric as reference
    
    # Try some random modifications
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        cell = [row[:] for row in base_cell]
        # Randomly flip some pixels in fundamental domain
        for _ in range(random.randint(1, 5)):
            i = random.randint(0, 15)
            j = random.randint(i, 15)  # j >= i for triangle
            # Flip this pixel and its symmetric counterparts
            old_val = cell[i][j]
            new_val = 1 - old_val
            # Apply to all symmetric positions
            positions = [(i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)]
            # Also transpose positions
            positions_set = set()
            for pi, pj in positions:
                positions_set.add((pi, pj))
                positions_set.add((pj, pi))  # transpose
            for pi, pj in positions_set:
                cell[pi][pj] = new_val
        
        # Verify symmetry
        if is_valid(cell):
            candidates.append(cell)
    
    # Try ring-based patterns
    for pattern_seed in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(pattern_seed)
        rings = [random.randint(0, 1) for _ in range(16)]
        # Ensure some structure: not all same
        if all(r == rings[0] for r in rings):
            continue
        cell = make_ring_pattern(rings)
        if is_valid(cell):
            candidates.append(cell)
    
    # For a proper evaluation, we'd need to compute the actual band structure.
    # Since we can't do that efficiently here, let's use a heuristic:
    # The optimal dielectric fraction for TM gaps in square lattice is around 50%
    # and the shape should be compact (circle-like) for the 4-5 gap.
    
    # Score candidates by dielectric fraction closeness to 50% and compactness
    def score_candidate(cell):
        frac = sum(sum(row) for row in cell) / (size * size)
        # Penalize deviation from 0.5
        frac_score = 1.0 - abs(frac - 0.5) * 2.0
        
        # Compute compactness: fraction of dielectric pixels that are near center
        cx = cy = (size - 1) / 2.0
        center_score = 0
        total = 0
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    dist = ((i - cx)**2 + (j - cy)**2) ** 0.5
                    center_score += max(0, 1.0 - dist / 15.0)
                    total += 1
        if total > 0:
            center_score /= total
        else:
            center_score = 0
        
        return 0.6 * frac_score + 0.4 * center_score
    
    scored = [(score_candidate(c), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])
    
    # Return the best candidate
    best_cell = scored[0][1]
    
    # Final validation
    assert is_valid(best_cell), "Symmetry check failed"
    assert len(best_cell) == size and len(best_cell[0]) == size
    
    return best_cell
# EVOLVE-BLOCK-END
