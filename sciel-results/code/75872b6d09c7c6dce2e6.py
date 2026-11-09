"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the unit cell as a binary mask defined by the union of four 
       rotated squares (diamonds) centered at the quarter-points, plus an optional 
       central square background.
    2. Optimize orientation angle, side length, and central square size via grid search.
    3. Use direct pixel-level simulated annealing as a secondary refinement.
    4. Enforce C4v symmetry throughout.
    5. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # Enforce C4v symmetry exactly
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                # All 8 symmetric points
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    # --- 2. Fast Evaluator ---
    # Use 7x7 plane waves (49 modes) for speed
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    # FFT index lookup
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # Along Gamma-M: (0, t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    # Along M-X: (t, 0.5-t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    # X (0.5, 0.5)
    k_points.append((0.5, 0.5))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        # Build epsilon grid
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues (we need indices 3 and 4 for bands 4 and 5)
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 1. Parametric Design: Union of Four Rotated Squares ---
    # Centers at quarter-points: (8,8), (8,24), (24,8), (24,24)
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    
    def generate_parametric_cell(angle_deg, side_len, central_size):
        """
        Generate a cell from parametric description:
        - angle_deg: rotation angle of the squares (degrees)
        - side_len: side length of each square
        - central_size: size of central square (0 if none)
        """
        cell = np.zeros((size, size), dtype=int)
        angle_rad = math.radians(angle_deg)
        
        # Generate the four rotated squares
        for cy, cx in centers:
            # Square vertices relative to center
            half_side = side_len / 2.0
            vertices = [
                (-half_side, -half_side),
                (half_side, -half_side),
                (half_side, half_side),
                (-half_side, half_side)
            ]
            
            # Rotate vertices
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            rotated_vertices = []
            for vx, vy in vertices:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                rotated_vertices.append((rx, ry))
            
            # Fill the square using point-in-polygon test
            for i in range(size):
                for j in range(size):
                    # Check if point (i, j) is inside the square
                    px = i - cy
                    py = j - cx
                    
                    # Point-in-convex-polygon test
                    inside = True
                    for k in range(4):
                        x1 = rotated_vertices[k][0]
                        y1 = rotated_vertices[k][1]
                        x2 = rotated_vertices[(k + 1) % 4][0]
                        y2 = rotated_vertices[(k + 1) % 4][1]
                        
                        # Cross product to check if point is on the same side
                        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                        if cross < 0:
                            inside = False
                            break
                    
                    if inside:
                        cell[i, j] = 1
        
        # Add central square if specified
        if central_size > 0:
            center = size // 2
            half_central = central_size / 2.0
            for i in range(size):
                for j in range(size):
                    if abs(i - center + 0.5) <= half_central and abs(j - center + 0.5) <= half_central:
                        cell[i, j] = 1
        
        return enforce_symmetry(cell)

    # --- 3. Grid Search over Parametric Space ---
    best_cell = None
    best_score = -float('inf')
    
    # Grid of angles, side lengths, and central sizes
    angles = [0, 15, 30, 45, 60, 75]
    side_lengths = [3, 4, 5, 6, 7, 8, 9, 10]
    central_sizes = [0, 2, 4, 6, 8]
    
    for angle in angles:
        for side in side_lengths:
            for central in central_sizes:
                if time.time() > deadline - 60.0:
                    break  # Leave time for refinement
                
                cell = generate_parametric_cell(angle, side, central)
                score = evaluate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
        
        if time.time() > deadline - 60.0:
            break

    if best_cell is None:
        # Fallback to simple disk configuration
        best_cell = np.zeros((size, size), dtype=int)
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < 4.0:
                        best_cell[i, j] = 1
        best_cell = enforce_symmetry(best_cell)
        best_score = evaluate_gap(best_cell)

    # --- 4. Secondary Refinement: Simulated Annealing ---
    current_cell = best_cell.copy()
    current_score = best_score
    
    temperature = 0.3
    cooling_rate = 0.995
    max_iterations = 30000
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Pick a random symmetric orbit to flip
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        orbit = [
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, j),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i)
        ]
        # Remove duplicates
        orbit = list(set(orbit))
        
        # Create new cell by flipping the orbit
        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = evaluate_gap(new_cell)
        
        # Accept or reject based on Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    # --- 5. Finalize ---
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
