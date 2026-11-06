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
    deadline = start_time + 100  # Safety margin

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

    def approximate_gap_ratio(cell, max_order=5, g_range=3):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
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

    def generate_separable_cell(x_pattern, y_pattern):
        """
        Generate a 32x32 cell from two 1D patterns of length 32.
        cell[i][j] = x_pattern[i] AND y_pattern[j]
        This ensures D4 symmetry if both patterns are palindromic.
        """
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if x_pattern[i] and y_pattern[j]:
                    cell[i][j] = 1
        return cell

    def generate_block_cell(x_bits, y_bits):
        """
        Generate a 32x32 cell from 4-bit patterns for x and y axes.
        Each bit controls an 8-pixel block. Bit=1 means dielectric.
        cell[i][j] = x_bits[i//8] AND y_bits[j//8]
        """
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            xb = x_bits[i // 8]
            for j in range(size):
                yb = y_bits[j // 8]
                cell[i][j] = xb & yb
        return cell

    # Strategy: Exhaustively evaluate all 256 separable block patterns
    # x_bits and y_bits are 4-bit patterns (0-15)
    best_score = -1.0
    best_cell = None
    best_params = None
    
    for x_bits in range(16):
        for y_bits in range(16):
            if time.time() - start_time > deadline:
                break
            cell = generate_block_cell(x_bits, y_bits)
            cell_sym = enforce_d4_symmetry(cell)
            
            # Skip empty or full cells
            total = sum(sum(row) for row in cell_sym)
            if total == 0 or total == size * size:
                continue
            
            score = approximate_gap_ratio(cell_sym)
            if score > best_score:
                best_score = score
                best_cell = cell_sym
                best_params = (x_bits, y_bits)
    
    # Also try the AND-OR variant: cell[i][j] = x_bits[i//8] | y_bits[j//8]
    for x_bits in range(16):
        for y_bits in range(16):
            if time.time() - start_time > deadline:
                break
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                xb = x_bits[i // 8]
                for j in range(size):
                    yb = y_bits[j // 8]
                    cell[i][j] = xb | yb
            cell_sym = enforce_d4_symmetry(cell)
            
            total = sum(sum(row) for row in cell_sym)
            if total == 0 or total == size * size:
                continue
            
            score = approximate_gap_ratio(cell_sym)
            if score > best_score:
                best_score = score
                best_cell = cell_sym
                best_params = (x_bits, y_bits, 'OR')
    
    # Also try XOR variant
    for x_bits in range(16):
        for y_bits in range(16):
            if time.time() - start_time > deadline:
                break
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                xb = x_bits[i // 8]
                for j in range(size):
                    yb = y_bits[j // 8]
                    cell[i][j] = xb ^ yb
            cell_sym = enforce_d4_symmetry(cell)
            
            total = sum(sum(row) for row in cell_sym)
            if total == 0 or total == size * size:
                continue
            
            score = approximate_gap_ratio(cell_sym)
            if score > best_score:
                best_score = score
                best_cell = cell_sym
                best_params = (x_bits, y_bits, 'XOR')
    
    # If no valid cell found, use a default
    if best_cell is None:
        # Default: central cross
        best_cell = [[0] * size for _ in range(size)]
        cx = size // 2
        cy = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - cx) <= 3 or abs(j - cy) <= 3:
                    best_cell[i][j] = 1
        best_cell = enforce_d4_symmetry(best_cell)
        best_score = approximate_gap_ratio(best_cell)
    
    # Local search: try flipping individual pixels in D4-symmetric orbits
    if time.time() - start_time < deadline - 30:
        base = best_cell
        improved = True
        max_rounds = 5
        round_num = 0
        
        while improved and round_num < max_rounds and time.time() - start_time < deadline - 30:
            improved = False
            round_num += 1
            
            # Try flipping each D4 orbit
            for i in range(size // 2):
                for j in range(size // 2):
                    if time.time() - start_time > deadline:
                        break
                    
                    # Get the D4 orbit of (i, j)
                    orbit = set()
                    for (ii, jj) in [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                                     (j, i), (size-1-j, size-1-i), (j, size-1-i), (size-1-j, i)]:
                        orbit.add((ii, jj))
                    
                    # Skip if orbit has been tried before or is at boundary
                    # Try flipping the entire orbit
                    cell_trial = [row[:] for row in base]
                    for (ii, jj) in orbit:
                        cell_trial[ii][jj] = 1 - base[ii][jj]
                    
                    cell_sym = enforce_d4_symmetry(cell_trial)
                    
                    if cell_sym != base:
                        score = approximate_gap_ratio(cell_sym)
                        if score > best_score:
                            best_score = score
                            best_cell = cell_sym
                            base = best_cell
                            improved = True
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
