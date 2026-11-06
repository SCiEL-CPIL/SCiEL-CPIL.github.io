"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 200s limit

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: row/col reversal and transpose invariance."""
        final_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[size-1-i][j],
                    cell[i][size-1-j],
                    cell[size-1-i][size-1-j],
                    cell[j][i],
                    cell[size-1-j][size-1-i],
                    cell[j][size-1-i],
                    cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                final_cell[i][j] = 1 if avg >= 0.5 else 0
        return final_cell

    def compute_fourier_coefficients(cell, max_order):
        """Compute Fourier coefficients of 1/epsilon using FFT for speed."""
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        # Use 2D FFT to get all coefficients at once
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                # Map negative indices to positive using FFT convention
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def fast_gap_ratio(cell, max_order=4, g_range=2, n_kpoints=3):
        """
        Very fast approximate gap ratio for local search.
        Uses fewer k-points and smaller G-vector basis.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors in the basis
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # Fewer k-points for speed
        if n_kpoints == 3:
            k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
        elif n_kpoints == 5:
            k_points = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5)]
        else:
            k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        eps_inv_G = 0.0
                    H[i, j] = k_sq * eps_inv_G
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def get_fundamental_domain_pixels(size):
        """
        Get pixels in the fundamental domain of D4 symmetry.
        For a 32x32 cell, the fundamental domain is the upper-left 8x8 quadrant
        (with appropriate boundary handling for even size).
        Actually for D4 of a square, the fundamental domain is 1/8 of the cell.
        For 32x32, that's pixels where i < j and i < 16 and j < 16... 
        More precisely: i in [0,15], j in [i, 15] gives the fundamental triangle.
        But we need to be careful about the exact symmetry group.
        
        The D4 group has 8 elements. The fundamental domain has size*size/8 = 128 pixels.
        For 32x32: i in [0, 15], j in [i, 15] gives 8*9/2 = 36... that's not right.
        
        Let me think more carefully. The D4 symmetry about center means:
        - (i,j) -> (size-1-i, j) [vertical flip]
        - (i,j) -> (i, size-1-j) [horizontal flip]  
        - (i,j) -> (j, i) [transpose]
        and all compositions.
        
        The fundamental domain is pixels where i <= j and i <= size-1-i and j <= size-1-j
        i.e., i <= j, i <= 15, j <= 16 for size=32.
        So i in [0,15], j in [i, 16]... but j max is 31, and j <= 16 means j in [i, 16].
        Wait, size-1-j >= j means j <= 15.5, so j <= 15 for integer j.
        And size-1-i >= i means i <= 15.5, so i <= 15.
        And i <= j.
        So fundamental domain: i in [0,15], j in [i, 15]. That's sum_{i=0}^{15} (15-i+1) = 128 pixels.
        32*32/8 = 128. Correct!
        """
        pixels = []
        half = size // 2  # 16 for size 32
        for i in range(half):
            for j in range(i, half):
                pixels.append((i, j))
        return pixels

    def approximate_gap_ratio(cell, max_order=5, g_range=3):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors in the basis
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # K-points in irreducible BZ: (0,0) to (0.5,0) to (0.5,0.5)
        k_points = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5)]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        eps_inv_G = 0.0
                    H[i, j] = k_sq * eps_inv_G
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def generate_single_rod(radius_frac):
        """Central circular rod."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    cell[i][j] = 1
        return cell

    def generate_ring(inner_frac, outer_frac):
        """Circular ring (dielectric annulus)."""
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                if r_in * r_in <= r2 < r_out * r_out:
                    cell[i][j] = 1
        return cell

    def generate_four_rods(radius_frac):
        """Four circular rods at corners (D4 symmetric)."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        # Rod centers at (size/4, size/4) and symmetric positions
        centers = [(size/4.0, size/4.0), (size*3/4.0, size/4.0),
                   (size/4.0, size*3/4.0), (size*3/4.0, size*3/4.0)]
        for i in range(size):
            for j in range(size):
                for cx, cy in centers:
                    x = (j + 0.5) - cx
                    y = (i + 0.5) - cy
                    if x * x + y * y < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_cross(arm_width_frac, arm_length_frac):
        """Cross shape (plus sign)."""
        cell = [[0] * size for _ in range(size)]
        w = int(arm_width_frac * size)
        l = int(arm_length_frac * size)
        cx = size // 2
        cy = size // 2
        half_w = w // 2
        for i in range(size):
            for j in range(size):
                dx = abs(j - cx)
                dy = abs(i - cy)
                if (dx <= half_w and dy <= l) or (dy <= half_w and dx <= l):
                    cell[i][j] = 1
        return cell

    def generate_2x2_supercell(rod_radius_frac):
        """
        Generate a 2x2 supercell arrangement of rods within the 32x32 unit cell.
        This creates four smaller rods arranged in a 2x2 pattern, which can open
        higher-order band gaps.
        """
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        # Centers of the four sub-rods in a 2x2 arrangement
        # Offset from center by size/4
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_2x2_supercell_ring(inner_frac, outer_frac):
        """
        Generate a 2x2 supercell arrangement of rings within the 32x32 unit cell.
        """
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    r2 = dx * dx + dy * dy
                    if r_in * r_in <= r2 < r_out * r_out:
                        cell[i][j] = 1
                        break
        return cell

    def generate_hexagonal_like(rod_radius_frac, spacing_frac):
        """
        Generate a pattern with rods arranged in a hexagonal-like D4 symmetric pattern.
        """
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        s = spacing_frac * size
        
        # Center rod
        centers = [(cx, cy)]
        # Six surrounding rods in hexagonal pattern, but adjust for D4 symmetry
        # Use 8 positions for D4 symmetry
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = math.radians(angle)
            cx_r = cx + s * math.cos(rad)
            cy_r = cy + s * math.sin(rad)
            centers.append((cx_r, cy_r))
            
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx_r, cy_r in centers:
                    dx = x - cx_r
                    dy = y - cy_r
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_diamond_pattern(arm_width_frac, size_frac):
        """Diamond (rotated square) pattern with D4 symmetry."""
        cell = [[0] * size for _ in range(size)]
        w = int(arm_width_frac * size)
        s = int(size_frac * size)
        cx = size / 2.0
        cy = size / 2.0
        half_w = w / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                # Diamond: |x|/a + |y|/b <= 1
                if abs(x) / s + abs(y) / s <= 1.0:
                    # Check if within the arm width from the diamond edge
                    dist_from_edge = 1.0 - (abs(x) / s + abs(y) / s)
                    if dist_from_edge * s >= half_w or abs(x) / s + abs(y) / s <= 0.3:
                        cell[i][j] = 1
        return cell

    def generate_checkerboard_block(block_size):
        """Block pattern with D4 symmetry."""
        cell = [[0] * size for _ in range(size)]
        bs = block_size
        for i in range(size):
            for j in range(size):
                # Map to block coordinates
                bi = i // bs
                bj = j // bs
                # D4 symmetric pattern: fill blocks where (bi+bj) is even
                if (bi + bj) % 2 == 0:
                    cell[i][j] = 1
        return cell

    def generate_concentric_rings(radii_fracs):
        """Multiple concentric rings."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                for k in range(len(radii_fracs) - 1):
                    r_in = radii_fracs[k] * size
                    r_out = radii_fracs[k + 1] * size
                    if r_in * r_in <= r2 < r_out * r_out:
                        cell[i][j] = 1
                        break
        return cell

    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.10, 0.25), (0.20, 0.30)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    # Crosses
    for w, l in [(0.15, 0.40), (0.20, 0.35), (0.25, 0.30), (0.15, 0.35)]:
        candidates.append(generate_cross(w, l))
    
    # Checkerboard blocks
    for bs in [4, 6, 8]:
        candidates.append(generate_checkerboard_block(bs))
    
    # Concentric rings
    for radii in [(0.0, 0.20), (0.0, 0.25, 0.40), (0.15, 0.30, 0.45)]:
        candidates.append(generate_concentric_rings(radii))
    
    # 2x2 supercell arrangements (key improvement based on knowledge)
    for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]:
        candidates.append(generate_2x2_supercell(r))
    
    # 2x2 supercell with rings
    for inner, outer in [(0.05, 0.15), (0.08, 0.18), (0.10, 0.20), (0.06, 0.16), 
                         (0.04, 0.12), (0.07, 0.14), (0.09, 0.17), (0.11, 0.19)]:
        candidates.append(generate_2x2_supercell_ring(inner, outer))
    
    # Additional 2x2 supercell variants with different offsets
    def generate_2x2_supercell_offset(rod_radius_frac, offset_frac):
        """2x2 supercell with adjustable offset from center."""
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        offset = offset_frac * size
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell
    
    for r, off in [(0.12, 0.25), (0.14, 0.25), (0.16, 0.25), (0.10, 0.30), 
                   (0.12, 0.30), (0.14, 0.30), (0.10, 0.20), (0.12, 0.20)]:
        candidates.append(generate_2x2_supercell_offset(r, off))
    
    # 2x2 supercell with square holes (inverse pattern)
    def generate_2x2_supercell_inverse(rod_radius_frac, hole_radius_frac):
        """2x2 supercell: dielectric background with air holes."""
        cell = [[1] * size for _ in range(size)]
        r_hole = hole_radius_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_hole * r_hole:
                        cell[i][j] = 0
                        break
        return cell
    
    for r_hole in [0.05, 0.08, 0.10, 0.12, 0.15]:
        candidates.append(generate_2x2_supercell_inverse(0.15, r_hole))
    
    # Hexagonal-like patterns
    for r, s in [(0.08, 0.25), (0.10, 0.25), (0.12, 0.30), (0.08, 0.30), (0.10, 0.30)]:
        candidates.append(generate_hexagonal_like(r, s))
    
    # Diamond patterns
    for w, s in [(0.15, 0.40), (0.20, 0.35), (0.15, 0.30), (0.25, 0.40)]:
        candidates.append(generate_diamond_pattern(w, s))
    
    # Evaluate all candidates using fast evaluator first, then refine top ones
    scored_candidates = []
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        # Use fast evaluator for initial screening
        fast_score = fast_gap_ratio(cell_sym, max_order=4, g_range=2, n_kpoints=3)
        scored_candidates.append((fast_score, cell_sym))
    
    # Sort by fast score, take top candidates for detailed evaluation
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = -1.0
    best_cell = candidates[0]
    
    # Evaluate top 10 with more accurate method
    top_n = min(10, len(scored_candidates))
    for fast_s, cell_sym in scored_candidates[:top_n]:
        if time.time() - start_time > deadline - 20:
            break
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # If time remains, try local perturbations around best design
    if time.time() - start_time < deadline - 15:
        base = best_cell
        improved = True
        max_iterations = 50
        iteration = 0
        
        # Get fundamental domain pixels for efficient search
        fd_pixels = get_fundamental_domain_pixels(size)
        
        while improved and time.time() - start_time < deadline - 10 and iteration < max_iterations:
            iteration += 1
            improved = False
            best_local = best_score
            best_local_cell = base
            
            # Try flipping individual pixels in fundamental domain only
            for fi, fj in fd_pixels:
                if time.time() - start_time > deadline - 5:
                    break
                
                # Try flipping single pixel (with D4 symmetry applied)
                cell_trial = [row[:] for row in base]
                cell_trial[fi][fj] = 1 - base[fi][fj]
                cell_sym = enforce_d4_symmetry(cell_trial)
                if cell_sym != base:
                    score = fast_gap_ratio(cell_sym, max_order=4, g_range=2, n_kpoints=3)
                    if score > best_local + 1e-6:
                        # Verify with more accurate evaluator
                        score_acc = approximate_gap_ratio(cell_sym)
                        if score_acc > best_local + 1e-6:
                            best_local = score_acc
                            best_local_cell = cell_sym
                            improved = True
                
                # Try flipping 2x2 patch centered at fundamental domain pixel
                cell_trial2 = [row[:] for row in base]
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = fi + di
                        nj = fj + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            cell_trial2[ni][nj] = 1 - base[ni][nj]
                cell_sym2 = enforce_d4_symmetry(cell_trial2)
                if cell_sym2 != base:
                    score2 = fast_gap_ratio(cell_sym2, max_order=4, g_range=2, n_kpoints=3)
                    if score2 > best_local + 1e-6:
                        score2_acc = approximate_gap_ratio(cell_sym2)
                        if score2_acc > best_local + 1e-6:
                            best_local = score2_acc
                            best_local_cell = cell_sym2
                            improved = True
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base = best_cell
    
    # Final polish: if time remains, do a more thorough local search
    if time.time() - start_time < deadline - 5:
        base = best_cell
        fd_pixels = get_fundamental_domain_pixels(size)
        
        # Try all single-pixel flips with accurate evaluator
        for fi, fj in fd_pixels:
            if time.time() - start_time > deadline - 2:
                break
            cell_trial = [row[:] for row in base]
            cell_trial[fi][fj] = 1 - base[fi][fj]
            cell_sym = enforce_d4_symmetry(cell_trial)
            if cell_sym != base:
                score = approximate_gap_ratio(cell_sym)
                if score > best_score + 1e-6:
                    best_score = score
                    best_cell = cell_sym
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
