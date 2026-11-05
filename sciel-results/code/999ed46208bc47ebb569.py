import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_symmetry(size, base):
    """
    Enforces full C4v symmetry (square symmetry group).
    Input: base (size x size array of 0/1)
    Output: cell (size x size list of lists of 0/1)
    """
    b = np.array(base, dtype=np.int8)
    N = size
    cell = np.zeros((N, N), dtype=np.int8)
    processed = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            
            # Generate the 8 symmetric positions for (i, j)
            orbit = [
                (i, j),
                (N-1-i, j),
                (j, i),
                (N-1-i, N-1-j),
                (i, N-1-j),
                (j, N-1-i),
                (N-1-j, i),
                (N-1-j, N-1-i)
            ]
            
            # Remove duplicates
            unique_orbit = list(set(orbit))
            
            # Majority vote from input base
            vals = [b[r, c] for r, c in unique_orbit]
            if sum(vals) >= len(vals) / 2.0:
                val = 1
            else:
                val = 0
            
            # Assign value to all unique positions in the orbit
            for r, c in unique_orbit:
                cell[r, c] = val
                processed[r, c] = True
                
    return cell.tolist()

def _compute_surrogate_gap(cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=12.0, eps_low=1.0):
    """
    Fast plane-wave TM band calculation surrogate.
    """
    # Build reciprocal lattice vectors
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    nG = len(Gs)
    
    # Compute inv_eps Fourier coefficients using FFT
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    inv_eps_fft = np.fft.fft2(inv_eps_arr)
    
    # Build inv_eps matrix
    dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
    dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    
    diff_bins_row = dmx % size
    diff_bins_col = dmy % size
    
    inv_eps_mat = inv_eps_fft[diff_bins_row, diff_bins_col]
    
    # K-points: Irreducible BZ of square lattice
    k_points = []
    k_points.append((0.0, 0.0))
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((t * np.pi, 0.0))
    for t in np.linspace(0.1, 1.0, n_ky):
        k_points.append((t * np.pi, t * np.pi))
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((np.pi, t * np.pi))
    
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k + Gs
        
        kGs_x = kGs[:, 0]
        kGs_y = kGs[:, 1]
        dot = np.outer(kGs_x, kGs_x) + np.outer(kGs_y, kGs_y)
        
        A = dot * inv_eps_mat
        A = (A + A.conj().T) / 2.0
        
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) >= 5:
            b4 = eigvals[3]
            b5 = eigvals[4]
            max_band4 = max(max_band4, b4)
            min_band5 = min(min_band5, b5)
    
    if min_band5 == np.inf or max_band4 == 0:
        return -1.0
    
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap

def _make_quadrant_rod_cell(size, center_offset, rod_radius, gap_width, background=1):
    """
    Create a cell with four quadrant rods (2x2 periodicity).
    center_offset: distance from cell center to each rod center
    rod_radius: radius of each rod
    gap_width: width of the gap between rods
    background: 1 for dielectric background, 0 for air background
    """
    N = size
    cell = np.full((N, N), background, dtype=np.int8)
    center = N // 2
    
    # Four rod centers at (±center_offset, 0) and (0, ±center_offset)
    rod_centers = [
        (center + center_offset, center),
        (center - center_offset, center),
        (center, center + center_offset),
        (center, center - center_offset)
    ]
    
    hole_val = 1 - background
    
    for i in range(N):
        for j in range(N):
            for rc_x, rc_y in rod_centers:
                dx = i - rc_x
                dy = j - rc_y
                if dx*dx + dy*dy <= rod_radius*rod_radius:
                    cell[i, j] = hole_val
                    break
    
    return _enforce_symmetry(N, cell.tolist())

def _make_concentric_ring_cell(size, inner_radius, outer_radius, background=1):
    """
    Create a cell with concentric ring structure.
    """
    N = size
    cell = np.full((N, N), background, dtype=np.int8)
    center = N // 2
    ring_val = 1 - background
    
    for i in range(N):
        for j in range(N):
            dx = i - center
            dy = j - center
            r2 = dx*dx + dy*dy
            if inner_radius*inner_radius <= r2 <= outer_radius*outer_radius:
                cell[i, j] = ring_val
    
    return _enforce_symmetry(N, cell.tolist())

def _make_cross_cell(size, arm_width, arm_length, background=0):
    """
    Create a cross-shaped dielectric structure.
    """
    N = size
    cell = np.full((N, N), background, dtype=np.int8)
    center = N // 2
    cross_val = 1 - background
    
    for i in range(N):
        for j in range(N):
            dx = abs(i - center)
            dy = abs(j - center)
            if (dx < arm_length and dy < arm_width/2.0) or \
               (dy < arm_length and dx < arm_width/2.0):
                cell[i, j] = cross_val
    
    return _enforce_symmetry(N, cell.tolist())

def _generate_candidates(size):
    """Generate diverse candidate cells."""
    candidates = []
    
    # 1. Quadrant rods with 2x2 periodicity
    for offset in [4.0, 5.0, 6.0, 7.0, 8.0]:
        for radius in [1.5, 2.0, 2.5, 3.0, 3.5]:
            candidates.append(_make_quadrant_rod_cell(size, offset, radius, 0, background=1))
    
    # 2. Concentric rings
    for inner in [3.0, 4.0, 5.0]:
        for outer in [6.0, 7.0, 8.0, 9.0]:
            if outer > inner:
                candidates.append(_make_concentric_ring_cell(size, inner, outer, background=1))
    
    # 3. Cross structures
    for width in [2.0, 3.0, 4.0]:
        for length in [6.0, 8.0, 10.0, 12.0]:
            candidates.append(_make_cross_cell(size, width, length, background=0))
    
    # 4. Single central hole (baseline)
    for r in [3.0, 4.0, 5.0, 6.0]:
        N = size
        cell = np.ones((N, N), dtype=np.int8)
        center = N // 2
        for i in range(N):
            for j in range(N):
                dx = i - center
                dy = j - center
                if dx*dx + dy*dy <= r*r:
                    cell[i, j] = 0
        candidates.append(_enforce_symmetry(N, cell.tolist()))
    
    return candidates

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap
    between the 4th and 5th bands.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s budget
    
    # Generate initial diverse candidates
    candidates = _generate_candidates(size)
    
    best_cell = None
    best_score = -1.0
    
    # Evaluate initial candidates
    for cell in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_surrogate_gap(cell, size, n_pw=8, n_kx=5, n_ky=5,
                                           eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search refinement
    if best_cell is not None:
        current_cell = best_cell
        current_score = best_score
        rng = np.random.default_rng(123)
        
        for round_num in range(8):
            if time.time() > deadline:
                break
            improved = False
            for step in range(30):
                if time.time() > deadline:
                    break
                # Random mutation with varying strength
                mut_cell = np.array(current_cell, dtype=np.int8)
                n_flips = rng.integers(1, 5)
                for _ in range(n_flips):
                    i = rng.integers(0, size)
                    j = rng.integers(0, size)
                    mut_cell[i, j] = 1 - mut_cell[i, j]
                sym_cell = _enforce_symmetry(size, mut_cell)
                
                try:
                    score = _compute_surrogate_gap(sym_cell, size, n_pw=8, n_kx=5, n_ky=5,
                                                   eps_high=eps_high, eps_low=eps_low)
                except Exception:
                    score = -1.0
                
                if score > current_score:
                    current_cell = sym_cell
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_cell = sym_cell
            
            if not improved:
                break
    
    # Final fallback
    if best_cell is None:
        N = size
        center = N // 2
        cell = np.ones((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(N):
                dx = i - center
                dy = j - center
                if dx*dx + dy*dy < 9:
                    cell[i, j] = 0
        best_cell = _enforce_symmetry(N, cell.tolist())
    
    return best_cell

# EVOLVE-BLOCK-END
