"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy: Use concentric square rings (Chebyshev distance based) with alternating
    dielectric/air assignments. This creates a structured microstructure that can
    open larger higher-order band gaps than simple rods. We evaluate candidate ring
    patterns using a fast plane-wave proxy and return the best one found.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (evaluator kills at 230s, need buffer)

    # Precompute Chebyshev distance map from center (0.5, 0.5) in normalized coords
    # For pixel (i, j), normalized coords are (i+0.5)/size - 0.5
    # Chebyshev distance: max(|x|, |y|)
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords)
    cheb_dist = np.maximum(np.abs(X), np.abs(Y))  # Shape (size, size)
    
    # The maximum Chebyshev distance in a 32x32 grid is about 0.5 (at corners)
    # Minimum is 0 (at center)
    # We'll define rings as intervals of cheb_dist
    
    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: row/col reversal and transpose invariance."""
        final_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i, j],
                    cell[size-1-i, j],
                    cell[i, size-1-j],
                    cell[size-1-i, size-1-j],
                    cell[j, i],
                    cell[size-1-j, size-1-i],
                    cell[j, size-1-i],
                    cell[size-1-j, i],
                ]
                avg = sum(vals) / len(vals)
                final_cell[i, j] = 1 if avg >= 0.5 else 0
        return final_cell

    def compute_fourier_coefficients(cell, max_order):
        """Compute Fourier coefficients of 1/epsilon using FFT for speed."""
        eps = np.array([[eps_high if cell[i, j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def evaluate_gap_ratio(cell, max_order=6, g_range=4):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        Returns the gap-midgap ratio: 2*(min_omega_5 - max_omega_4)/(min_omega_5 + max_omega_4)
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors in the basis
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # K-points in irreducible BZ
        k_points = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5),
                    (0.1, 0.0), (0.4, 0.0), (0.3, 0.3), (0.45, 0.05)]
        
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
        return 2.0 * gap / midgap

    def generate_ring_pattern(ring_thresholds, ring_values):
        """
        Generate a cell from concentric square rings.
        ring_thresholds: sorted list of Chebyshev distance thresholds
        ring_values: list of 0/1 values for each ring region
        """
        cell = np.zeros((size, size), dtype=int)
        prev_threshold = 0.0
        for idx, threshold in enumerate(ring_thresholds):
            if idx < len(ring_values):
                val = ring_values[idx]
            else:
                val = 0
            mask = (cheb_dist >= prev_threshold) & (cheb_dist < threshold)
            cell[mask] = val
            prev_threshold = threshold
        # Fill remaining (outermost)
        if len(ring_values) > len(ring_thresholds):
            val = ring_values[-1]
            mask = cheb_dist >= prev_threshold
            cell[mask] = val
        return cell

    # Define candidate ring patterns
    # The Chebyshev distance ranges from 0 to ~0.5
    # We'll try various numbers of rings and patterns
    
    best_score = -10.0
    best_cell = None
    
    # Precompute a few standard patterns for quick evaluation
    candidates = []
    
    # Pattern 1: Single central dielectric block (square)
    # Ring at 0 to 0.15
    candidates.append(([0.15], [1]))
    
    # Pattern 2: Central dielectric square + air ring + dielectric outer
    candidates.append(([0.10, 0.20, 0.50], [1, 0, 1]))
    
    # Pattern 3: Air center, dielectric ring, air outer
    candidates.append(([0.08, 0.18, 0.50], [0, 1, 0]))
    
    # Pattern 4: Alternating rings
    candidates.append(([0.06, 0.12, 0.18, 0.24, 0.50], [1, 0, 1, 0, 1]))
    
    # Pattern 5: Dense central dielectric
    candidates.append(([0.20], [1]))
    
    # Pattern 6: Moderate central dielectric
    candidates.append(([0.12], [1]))
    
    # Pattern 7: Two concentric dielectric rings
    candidates.append(([0.08, 0.16, 0.24, 0.50], [1, 0, 1, 0]))
    
    # Pattern 8: Dielectric frame (air center, thick dielectric ring)
    candidates.append(([0.12, 0.28, 0.50], [0, 1, 0]))
    
    # Pattern 9: Thin central dielectric
    candidates.append(([0.05], [1]))
    
    # Pattern 10: Thick central dielectric
    candidates.append(([0.25], [1]))
    
    # Evaluate candidates
    for ring_thresholds, ring_values in candidates:
        if time.time() > deadline:
            break
        cell = generate_ring_pattern(ring_thresholds, ring_values)
        cell = enforce_d4_symmetry(cell)
        score = evaluate_gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If we have time, try a small grid search over ring parameters
    if time.time() < deadline - 30 and best_cell is not None:
        # Try varying the number of rings and their positions
        for n_rings in [3, 4, 5]:
            if time.time() > deadline - 10:
                break
            # Generate evenly spaced ring thresholds
            for base_spacing in [0.04, 0.05, 0.06, 0.07, 0.08]:
                if time.time() > deadline - 10:
                    break
                thresholds = [base_spacing * (k + 1) for k in range(n_rings)]
                if thresholds[-1] > 0.45:
                    continue
                # Try alternating patterns
                for start_val in [0, 1]:
                    values = [start_val if k % 2 == 0 else 1 - start_val for k in range(n_rings)]
                    values.append(1 - values[-1])  # Outer region
                    cell = generate_ring_pattern(thresholds, values)
                    cell = enforce_d4_symmetry(cell)
                    score = evaluate_gap_ratio(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
    
    # If no good cell found, fall back to a simple rod
    if best_cell is None:
        # Central circular rod
        best_cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        r_px = 0.15 * size
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    best_cell[i, j] = 1
        best_cell = enforce_d4_symmetry(best_cell)
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
