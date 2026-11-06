"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, base_pattern):
    """
    Given a 1/4 cell (top-left quadrant), expand to full symmetric cell.
    """
    half = size // 2
    cell = [[0] * size for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = base_pattern[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
    return cell

def _precompute_tm_matrix(cell, size, n_pw):
    """
    Precompute the constant matrix M where M_{GG'} = (1/epsilon)_{G-G'}.
    This matrix is independent of k.
    """
    eps_low = 1.0
    eps_high = 12.0
    
    # Create 1/epsilon map
    inv_eps = np.ones((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                inv_eps[i][j] = 1.0 / eps_high
    
    # Compute Fourier coefficients of 1/epsilon
    # c_G = (1/N^2) sum_{x,y} inv_eps[x,y] exp(-i 2 pi (Gx x + Gy y) / N)
    # numpy fft2 does sum f exp(-i 2 pi k n / N). So F[k] = sum f exp(-i...).
    # c_G = F[mod(G, N)] / N^2
    F = np.fft.fft2(inv_eps)
    N = size
    coeff = np.zeros((n_pw, n_pw), dtype=np.complex128)
    
    half_n = (n_pw - 1) // 2
    G_indices = np.arange(-half_n, half_n + 1)
    
    for idx, gx in enumerate(G_indices):
        for idy, gy in enumerate(G_indices):
            kx = gx % N
            ky = gy % N
            coeff[idx][idy] = F[kx][ky] / (N * N)
            
    # Build M matrix of size N_pw * N_pw
    # M[row, col] = coeff[G_row - G_col]
    N_tot = n_pw * n_pw
    M = np.zeros((N_tot, N_tot), dtype=np.complex128)
    
    # Map G index to array index
    # G_indices is 1D, but we have 2D G vectors.
    # We need to map a 2D difference (dGx, dGy) to an index in coeff.
    # coeff is indexed by [idx_x, idx_y] where idx_x corresponds to Gx offset.
    
    for i in range(n_pw):
        for j in range(n_pw):
            row = i * n_pw + j
            Gx_i = G_indices[i]
            Gy_i = G_indices[j]
            
            for ip in range(n_pw):
                for jp in range(n_pw):
                    col = ip * n_pw + jp
                    Gx_j = G_indices[ip]
                    Gy_j = G_indices[jp]
                    
                    dGx = Gx_i - Gx_j
                    dGy = Gy_i - Gy_j
                    
                    # Check if dG is within the range of coeff
                    if -half_n <= dGx <= half_n and -half_n <= dGy <= half_n:
                        coeff_idx_x = dGx + half_n
                        coeff_idx_y = dGy + half_n
                        M[row][col] = coeff[coeff_idx_x][coeff_idx_y]
    
    return M, G_indices, n_pw

def _compute_gap_ratio_fast(cell, size, n_pw=17, n_bands=5):
    """
    Compute the gap-midgap ratio using pre-computed M.
    """
    try:
        M, G_indices, n_pw = _precompute_tm_matrix(cell, size, n_pw)
        N_tot = n_pw * n_pw
        half_n = (n_pw - 1) // 2
        
        # Precompute G vector squared magnitudes? No, (k+G)^2 depends on k.
        # But we can precompute Gx, Gy arrays for speed.
        Gx_arr = np.repeat(G_indices, n_pw)
        Gy_arr = np.tile(G_indices, n_pw)
        
        # Sample points in the irreducible Brillouin zone
        # IBZ for square lattice: 0 <= kx <= ky, kx + ky <= pi (in units of 2*pi/a? No, usually k is in 2*pi/a units or just radians?)
        # The problem says "30 wave vectors covering the boundary and interior".
        # Standard IBZ for square lattice (a=1): -pi/2 <= kx,ky <= pi/2.
        # Symmetry reduces to triangle: 0 <= kx <= ky, kx + ky <= pi/2?
        # Actually, the first BZ is [-pi, pi] x [-pi, pi] if we use G = 2*pi*n/a.
        # With a=1, G = 2*pi*n.
        # The IBZ is the triangle with vertices (0,0), (pi,0), (pi/2, pi/2)?
        # Let's use the standard IBZ for square lattice:
        # 0 <= kx <= ky, kx + ky <= pi. (This assumes k is in radians, and BZ edge is pi).
        # Wait, if a=1, G = 2*pi. The BZ boundary is at pi.
        # So IBZ: 0 <= kx <= ky, kx + ky <= pi.
        
        points = []
        # Gamma point
        points.append((0.0, 0.0))
        # M point (pi, 0) and (0, pi)
        points.append((math.pi, 0.0))
        points.append((0.0, math.pi))
        # X point (pi/2, pi/2)
        points.append((math.pi/2, math.pi/2))
        
        # Sample along boundaries and interior
        # kx = 0, ky from 0 to pi
        for ky in np.linspace(0.0, math.pi, 10):
            points.append((0.0, ky))
        # kx = ky, k from 0 to pi/2
        for k in np.linspace(0.0, math.pi/2, 10):
            points.append((k, k))
        # kx + ky = pi, kx from 0 to pi/2
        for kx in np.linspace(0.0, math.pi/2, 10):
            points.append((kx, math.pi - kx))
            
        # Remove duplicates and limit to 30
        seen = set()
        unique_points = []
        for p in points:
            key = (round(p[0], 6), round(p[1], 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)
        
        unique_points = unique_points[:30]
        
        max_band4 = 0.0
        min_band5 = float('inf')
        
        for kx, ky in unique_points:
            # Compute (k+G)^2
            kGx = kx + Gx_arr
            kGy = ky + Gy_arr
            kG_sq = kGx * kGx + kGy * kGy
            
            # H = M * diag(kG_sq)
            # This is equivalent to M * (kG_sq[:, None])
            H = M * kG_sq[:, None]
            
            # Solve eigenvalues
            # H is Hermitian
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                
                if len(eigenvalues) >= n_bands:
                    b4 = eigenvalues[3]
                    b5 = eigenvalues[4]
                    max_band4 = max(max_band4, b4)
                    min_band5 = min(min_band5, b5)
            except Exception:
                pass
        
        if min_band5 == float('inf') or max_band4 == 0.0:
            return 0.0
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap == 0:
            return 0.0
        
        return 2.0 * gap / midgap
    except Exception:
        return -1.0

def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0  # 30s margin
    
    size = problem["pixels"]
    
    def _eval_cell(cell):
        return _compute_gap_ratio_fast(cell, size, n_pw=17, n_bands=5)
    
    # Generate candidates
    candidates = []
    
    def _make_rod(radius, cx=0.5, cy=0.5):
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size
                y = (j + 0.5) / size
                # Mirror to center
                # We are in top-left quadrant. Center of cell is (0.5, 0.5) in cell coords.
                # In quadrant coords (0..0.5), center is (0.5, 0.5) relative to quadrant? No.
                # Let's just use cell coordinates for the check, but only fill top-left.
                # Actually, _make_symmetric_cell handles the mirroring.
                # So base[i][j] corresponds to cell[i][j].
                # Cell center is (15.5, 15.5) in pixel indices?
                # No, size=32. Center is between 15 and 16.
                # Let's use normalized coords x, y in [0, 1].
                # i=0 -> x=0.03125, i=15 -> x=0.46875.
                # The rod is centered at (0.5, 0.5).
                # Distance from (x, y) to (0.5, 0.5).
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < radius * radius:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)
    
    def _make_4_rods(radius, offset):
        # Rods at (0.5+offset, 0.5+offset) etc?
        # Due to symmetry, a rod at (x,y) in top-left quadrant implies rods at
        # (x,y), (x, 1-y), (1-x, y), (1-x, 1-y).
        # If we place a rod at (0.5+offset, 0.5+offset) in the FULL cell,
        # it will be in the top-right quadrant if offset>0?
        # Let's define the rod center in the top-left quadrant coordinates.
        # The top-left quadrant covers x in [0, 0.5], y in [0, 0.5].
        # A rod at (0.25, 0.25) in the full cell is at (0.25, 0.25) in the quadrant.
        # This creates 4 rods at (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75).
        base = [[0] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size
                y = (j + 0.5) / size
                # Center of this specific rod in the quadrant
                cx_q = 0.5 - offset # If offset is distance from center towards corner
                cy_q = 0.5 - offset
                # Wait, if offset=0, rod is at 0.5,0.5 (center of cell).
                # If offset=0.25, rod is at 0.25,0.25.
                dx = x - cx_q
                dy = y - cy_q
                if dx * dx + dy * dy < radius * radius:
                    base[i][j] = 1
        return _make_symmetric_cell(size, base)

    # Candidate 1: Single rod
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_rod(r))
        
    # Candidate 2: 4 rods (offset from center)
    # The knowledge suggests 4 rods at +/-a/4.
    # a=1. a/4 = 0.25.
    # So offset = 0.25 from center?
    # Center is 0.5. 0.5 - 0.25 = 0.25.
    # So rod center at 0.25, 0.25 in quadrant.
    for r in [0.05, 0.08, 0.1, 0.12, 0.15]:
        candidates.append(_make_4_rods(r, 0.25))
        
    # Candidate 3: Air holes (inverse)
    def _make_air_hole(radius, cx=0.5, cy=0.5):
        base = [[1] * (size // 2) for _ in range(size // 2)]
        for i in range(size // 2):
            for j in range(size // 2):
                x = (i + 0.5) / size
                y = (j + 0.5) / size
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < radius * radius:
                    base[i][j] = 0
        return _make_symmetric_cell(size, base)
        
    for r in [0.1, 0.2, 0.3]:
        candidates.append(_make_air_hole(r))
        
    # Evaluate candidates
    best_score = -1.0
    best_cell = None
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        score = _eval_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            print(f"Candidate {idx}: score={score:.6f}", flush=True)
            
    # Local Search on the best candidate
    if best_cell is not None and time.time() < deadline - 10:
        current_cell = [row[:] for row in best_cell]
        current_score = best_score
        half = size // 2
        
        # Iterate over pixels in top-left quadrant
        # To save time, we can do a few passes
        for pass_num in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(half):
                if time.time() > deadline - 5:
                    break
                for j in range(half):
                    if time.time() > deadline - 5:
                        break
                    old_val = current_cell[i][j]
                    current_cell[i][j] = 1 - old_val
                    # Propagate
                    current_cell[i][size - 1 - j] = current_cell[i][j]
                    current_cell[size - 1 - i][j] = current_cell[i][j]
                    current_cell[size - 1 - i][size - 1 - j] = current_cell[i][j]
                    
                    new_score = _eval_cell(current_cell)
                    
                    if new_score > current_score:
                        current_score = new_score
                        best_score = new_score
                        best_cell = [row[:] for row in current_cell]
                        improved = True
                    else:
                        # Revert
                        current_cell[i][j] = old_val
                        current_cell[i][size - 1 - j] = old_val
                        current_cell[size - 1 - i][j] = old_val
                        current_cell[size - 1 - i][size - 1 - j] = old_val
                        
            if not improved:
                break
                
    if best_cell is None:
        # Fallback to a simple rod
        best_cell = _make_rod(0.2)
        
    return best_cell

# EVOLVE-BLOCK-END
