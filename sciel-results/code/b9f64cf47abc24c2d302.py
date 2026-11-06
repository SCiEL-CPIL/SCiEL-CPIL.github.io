"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy: Solid dielectric background with air holes in centered-square
    lattice (primary holes at center, secondary holes at offset positions).
    This dual-structure approach explores high-contrast designs.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 200s limit

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
        Uses larger basis set for better accuracy.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # More k-points in irreducible BZ for better sampling
        k_points = [
            (0.0, 0.0), (0.125, 0.0), (0.25, 0.0), (0.375, 0.0), (0.5, 0.0),
            (0.125, 0.125), (0.25, 0.125), (0.375, 0.125),
            (0.125, 0.25), (0.25, 0.25), (0.375, 0.25),
            (0.125, 0.375), (0.25, 0.375), (0.375, 0.375),
            (0.0, 0.25), (0.25, 0.5), (0.5, 0.25), (0.5, 0.5)
        ]
        
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

    def generate_dual_hole_structure(primary_radius, secondary_radius, offset_frac, num_secondary=4):
        """
        Solid dielectric background with air holes:
        - Primary hole at center
        - Secondary holes at offset positions (D4 symmetric)
        Returns cell where 1=dielectric, 0=air
        """
        cell = [[1] * size for _ in range(size)]  # Start with all dielectric
        cx = size / 2.0
        cy = size / 2.0
        
        # Primary hole at center
        r_primary = primary_radius * size
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_primary * r_primary:
                    cell[i][j] = 0  # Air hole
        
        # Secondary holes at offset positions
        if secondary_radius > 0:
            r_secondary = secondary_radius * size
            offset = offset_frac * size
            # 4 positions for D4 symmetry: (±offset, 0) and (0, ±offset)
            secondary_centers = [
                (cx + offset, cy),
                (cx - offset, cy),
                (cx, cy + offset),
                (cx, cy - offset),
            ]
            for sx, sy in secondary_centers:
                for i in range(size):
                    for j in range(size):
                        x = (j + 0.5) - sx
                        y = (i + 0.5) - sy
                        if x * x + y * y < r_secondary * r_secondary:
                            cell[i][j] = 0  # Air hole
        
        return cell

    def generate_dual_hole_diagonal(primary_radius, secondary_radius, offset_frac):
        """
        Solid dielectric with primary center hole and diagonal secondary holes.
        Secondary holes at (±offset, ±offset) for D4 symmetry.
        """
        cell = [[1] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        r_primary = primary_radius * size
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_primary * r_primary:
                    cell[i][j] = 0
        
        if secondary_radius > 0:
            r_secondary = secondary_radius * size
            offset = offset_frac * size
            secondary_centers = [
                (cx + offset, cy + offset),
                (cx - offset, cy + offset),
                (cx + offset, cy - offset),
                (cx - offset, cy - offset),
            ]
            for sx, sy in secondary_centers:
                for i in range(size):
                    for j in range(size):
                        x = (j + 0.5) - sx
                        y = (i + 0.5) - sy
                        if x * x + y * y < r_secondary * r_secondary:
                            cell[i][j] = 0
        
        return cell

    def generate_dual_hole_8fold(primary_radius, secondary_radius, offset_frac):
        """
        Solid dielectric with primary center hole and 8 secondary holes
        at cardinal and diagonal positions for enhanced D4 symmetry.
        """
        cell = [[1] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        r_primary = primary_radius * size
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_primary * r_primary:
                    cell[i][j] = 0
        
        if secondary_radius > 0:
            r_secondary = secondary_radius * size
            offset = offset_frac * size
            secondary_centers = [
                (cx + offset, cy), (cx - offset, cy),
                (cx, cy + offset), (cx, cy - offset),
                (cx + offset, cy + offset), (cx - offset, cy + offset),
                (cx + offset, cy - offset), (cx - offset, cy - offset),
            ]
            for sx, sy in secondary_centers:
                for i in range(size):
                    for j in range(size):
                        x = (j + 0.5) - sx
                        y = (i + 0.5) - sy
                        if x * x + y * y < r_secondary * r_secondary:
                            cell[i][j] = 0
        
        return cell

    def generate_ring_hole_structure(inner_radius, outer_radius, hole_radius, offset_frac):
        """
        Dielectric ring with air holes at center and offset positions.
        """
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        r_in = inner_radius * size
        r_out = outer_radius * size
        r_hole = hole_radius * size
        
        # Dielectric ring
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                if r_in * r_in <= r2 < r_out * r_out:
                    cell[i][j] = 1
        
        # Air hole at center
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_hole * r_hole:
                    cell[i][j] = 0
        
        # Offset air holes
        offset = offset_frac * size
        hole_centers = [
            (cx + offset, cy), (cx - offset, cy),
            (cx, cy + offset), (cx, cy - offset),
        ]
        for hx, hy in hole_centers:
            for i in range(size):
                for j in range(size):
                    x = (j + 0.5) - hx
                    y = (i + 0.5) - hy
                    if x * x + y * y < r_hole * r_hole:
                        cell[i][j] = 0
        
        return cell

    # Generate candidate designs
    candidates = []
    
    # Dual hole structures - cardinal secondary holes
    for pr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for sr in [0.0, 0.08, 0.12, 0.15, 0.20]:
            for off in [0.15, 0.20, 0.25, 0.30, 0.35]:
                if sr == 0.0:
                    candidates.append(generate_dual_hole_structure(pr, 0, 0))
                else:
                    candidates.append(generate_dual_hole_structure(pr, sr, off))
    
    # Dual hole structures - diagonal secondary holes
    for pr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for sr in [0.08, 0.12, 0.15, 0.20]:
            for off in [0.15, 0.20, 0.25, 0.30]:
                candidates.append(generate_dual_hole_diagonal(pr, sr, off))
    
    # Dual hole structures - 8-fold secondary holes
    for pr in [0.10, 0.15, 0.20, 0.25]:
        for sr in [0.06, 0.08, 0.10, 0.12]:
            for off in [0.18, 0.22, 0.26, 0.30]:
                candidates.append(generate_dual_hole_8fold(pr, sr, off))
    
    # Ring-hole structures
    for inner, outer in [(0.15, 0.35), (0.20, 0.40), (0.25, 0.45)]:
        for hr in [0.05, 0.08, 0.10]:
            for off in [0.20, 0.25, 0.30]:
                candidates.append(generate_ring_hole_structure(inner, outer, hr, off))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # Local optimization around best design
    if time.time() - start_time < deadline - 20:
        base = best_cell
        improved = True
        max_iterations = 3
        iteration = 0
        
        while improved and time.time() - start_time < deadline - 10 and iteration < max_iterations:
            improved = False
            best_local = best_score
            best_local_cell = base
            iteration += 1
            
            # Try flipping pixels in a focused region around center
            for flip_i in range(8, size - 8):
                for flip_j in range(8, size - 8):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    cell_trial = [row[:] for row in base]
                    cell_trial[flip_i][flip_j] = 1 - base[flip_i][flip_j]
                    cell_sym = enforce_d4_symmetry(cell_trial)
                    if cell_sym != base:
                        score = approximate_gap_ratio(cell_sym)
                        if score > best_local + 1e-6:
                            best_local = score
                            best_local_cell = cell_sym
                            improved = True
                
                if time.time() - start_time > deadline - 5:
                    break
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base = best_cell
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
