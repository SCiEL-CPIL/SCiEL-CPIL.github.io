"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell with D4 symmetry
    to maximize the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 200s limit

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
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def approximate_gap_ratio(cell, max_order=6, g_range=4):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors in the basis
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # K-points in irreducible BZ (more points for better accuracy)
        k_points = [(0.0, 0.0), (0.0625, 0.0), (0.125, 0.0), (0.1875, 0.0), (0.25, 0.0),
                    (0.3125, 0.0), (0.375, 0.0), (0.4375, 0.0), (0.5, 0.0),
                    (0.0625, 0.0625), (0.125, 0.0625), (0.1875, 0.0625), (0.25, 0.0625),
                    (0.3125, 0.0625), (0.375, 0.0625), (0.4375, 0.0625), (0.5, 0.0625),
                    (0.0625, 0.125), (0.125, 0.125), (0.1875, 0.125), (0.25, 0.125),
                    (0.3125, 0.125), (0.375, 0.125), (0.4375, 0.125), (0.5, 0.125),
                    (0.0625, 0.1875), (0.125, 0.1875), (0.1875, 0.1875), (0.25, 0.1875),
                    (0.3125, 0.1875), (0.375, 0.1875), (0.4375, 0.1875), (0.5, 0.1875),
                    (0.0625, 0.25), (0.125, 0.25), (0.1875, 0.25), (0.25, 0.25),
                    (0.3125, 0.25), (0.375, 0.25), (0.4375, 0.25), (0.5, 0.25),
                    (0.0625, 0.3125), (0.125, 0.3125), (0.1875, 0.3125), (0.25, 0.3125),
                    (0.3125, 0.3125), (0.375, 0.3125), (0.4375, 0.3125), (0.5, 0.3125),
                    (0.0625, 0.375), (0.125, 0.375), (0.1875, 0.375), (0.25, 0.375),
                    (0.3125, 0.375), (0.375, 0.375), (0.4375, 0.375), (0.5, 0.375),
                    (0.0625, 0.4375), (0.125, 0.4375), (0.1875, 0.4375), (0.25, 0.4375),
                    (0.3125, 0.4375), (0.375, 0.4375), (0.4375, 0.4375), (0.5, 0.4375),
                    (0.125, 0.5), (0.1875, 0.5), (0.25, 0.5), (0.3125, 0.5),
                    (0.375, 0.5), (0.4375, 0.5), (0.5, 0.5)]
        
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

    def local_search_refine(cell, max_iterations=50, time_budget=30):
        """
        Local search by flipping pixels while maintaining D4 symmetry.
        Uses a soft surrogate to guide the search.
        """
        start_time = time.time()
        best_cell = [row[:] for row in cell]
        best_score = approximate_gap_ratio(best_cell)
        
        # Get D4 orbit representatives
        orbit_reps = []
        seen = set()
        for i in range(size):
            for j in range(size):
                if (i, j) in seen:
                    continue
                # Compute full orbit
                orbit = set()
                orbit.add((i, j))
                orbit.add((size-1-i, j))
                orbit.add((i, size-1-j))
                orbit.add((size-1-i, size-1-j))
                orbit.add((j, i))
                orbit.add((size-1-j, size-1-i))
                orbit.add((j, size-1-i))
                orbit.add((size-1-j, i))
                for pos in orbit:
                    seen.add(pos)
                orbit_reps.append(orbit)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_budget:
                break
            
            improved = False
            for orbit in orbit_reps:
                if time.time() - start_time > time_budget:
                    break
                
                # Try flipping all pixels in this orbit
                trial_cell = [row[:] for row in best_cell]
                for (i, j) in orbit:
                    trial_cell[i][j] = 1 - trial_cell[i][j]
                
                trial_score = approximate_gap_ratio(trial_cell)
                
                if trial_score > best_score:
                    best_cell = trial_cell
                    best_score = trial_score
                    improved = True
            
            if not improved:
                break
        
        return best_cell, best_score

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

    def generate_square_rod(size_frac):
        """Centered square rod."""
        cell = [[0] * size for _ in range(size)]
        s = int(size_frac * size)
        start = size // 2 - s // 2
        end = start + s
        for i in range(start, end):
            for j in range(start, end):
                cell[i][j] = 1
        return cell

    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]:
        candidates.append(generate_single_rod(r))
    
    # Rings with more combinations
    for inner, outer in [(0.05, 0.20), (0.10, 0.25), (0.15, 0.30), (0.20, 0.35), (0.25, 0.40),
                         (0.10, 0.20), (0.15, 0.25), (0.20, 0.30), (0.25, 0.35), (0.05, 0.15),
                         (0.10, 0.30), (0.15, 0.35), (0.20, 0.40)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22]:
        candidates.append(generate_four_rods(r))
    
    # Cross shapes
    for w_frac, l_frac in [(0.10, 0.35), (0.15, 0.40), (0.20, 0.45), (0.25, 0.50),
                           (0.12, 0.38), (0.18, 0.42), (0.22, 0.48), (0.08, 0.30)]:
        candidates.append(generate_cross(w_frac, l_frac))
    
    # Concentric rings
    candidates.append(generate_concentric_rings([0.0, 0.15, 0.30, 0.45]))
    candidates.append(generate_concentric_rings([0.10, 0.25, 0.40]))
    candidates.append(generate_concentric_rings([0.0, 0.10, 0.25, 0.40]))
    candidates.append(generate_concentric_rings([0.05, 0.20, 0.35]))
    candidates.append(generate_concentric_rings([0.0, 0.12, 0.28, 0.44]))
    
    # Square rods
    for s_frac in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        candidates.append(generate_square_rod(s_frac))
    
    # Enforce D4 symmetry on all candidates
    for idx in range(len(candidates)):
        candidates[idx] = enforce_d4_symmetry(candidates[idx])
    
    # Evaluate all candidates and find the best
    best_score = -1.0
    best_cell = candidates[0] if candidates else [[0] * size for _ in range(size)]
    
    # First pass: evaluate all candidates
    for idx, cell in enumerate(candidates):
        if time.time() > deadline:
            break
        score = approximate_gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Second pass: local search refinement on top candidates
    if time.time() < deadline - 60:
        # Sort candidates by score and refine top ones
        candidate_scores = []
        for cell in candidates:
            s = approximate_gap_ratio(cell)
            candidate_scores.append((s, cell))
        candidate_scores.sort(reverse=True)
        
        # Refine top 3 candidates
        for s, cell in candidate_scores[:3]:
            if time.time() > deadline - 30:
                break
            refined_cell, refined_score = local_search_refine(cell, max_iterations=30, time_budget=20)
            if refined_score > best_score:
                best_score = refined_score
                best_cell = refined_cell
    
    return best_cell
# EVOLVE-BLOCK-END
