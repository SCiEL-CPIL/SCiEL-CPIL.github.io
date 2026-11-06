"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _make_four_rod_cell(size, offset, radius, shape_param=0.0):
    """Generate a 4-rod symmetric cell.
    
    offset: distance of each rod center from cell center (0.2-0.35)
    radius: base radius of each rod (0.08-0.25)
    shape_param: controls squareness (0=circle, 1=square)
    """
    cell = np.zeros((size, size), dtype=int)
    half = size / 2.0
    
    # Four rod centers
    centers = [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                
                if shape_param < 0.5:
                    # Circular
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radius:
                        cell[i, j] = 1
                else:
                    # Superellipse: |dx/r|^p + |dy/r|^p < 1, p = 2 + 2*shape_param
                    p = 2.0 + 2.0 * shape_param
                    val = (abs(dx)/radius)**p + (abs(dy)/radius)**p
                    if val < 1.0:
                        cell[i, j] = 1
    
    return _apply_symmetry(cell).tolist()

def _make_cross_cell(size, arm_width, arm_length, corner_radius=0.0):
    """Generate a cross-shaped dielectric pattern."""
    cell = np.zeros((size, size), dtype=int)
    half = size / 2.0
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Horizontal arm
            if abs(y) < arm_width/2 and abs(x) < arm_length/2:
                cell[i, j] = 1
            # Vertical arm
            elif abs(x) < arm_width/2 and abs(y) < arm_length/2:
                cell[i, j] = 1
            # Corner fill if specified
            elif corner_radius > 0:
                dx = abs(x) - arm_length/2 + arm_width/2
                dy = abs(y) - arm_length/2 + arm_width/2
                if dx > 0 and dy > 0:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < corner_radius:
                        cell[i, j] = 1
    
    return _apply_symmetry(cell).tolist()

def _make_staggered_cell(size, n_rods_per_row, rod_radius, row_offset=0.0):
    """Generate a staggered array of rods with 4-fold symmetry."""
    cell = np.zeros((size, size), dtype=int)
    
    # Create a grid of rod positions with 4-fold symmetry
    # For n_rods_per_row = 2: rods at (±0.25, ±0.25)
    # For n_rods_per_row = 3: rods at (-0.33,0,0.33) in each direction
    spacing = 1.0 / (n_rods_per_row + 1)
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Check distance to nearest rod center
            best_dist = 1.0
            for ix in range(n_rods_per_row):
                for iy in range(n_rods_per_row):
                    cx = (ix + 0.5) * spacing - 0.5 + row_offset
                    cy = (iy + 0.5) * spacing - 0.5
                    dx = x - cx
                    dy = y - cy
                    dist2 = dx*dx + dy*dy
                    if dist2 < best_dist:
                        best_dist = dist2
            
            if np.sqrt(best_dist) < rod_radius:
                cell[i, j] = 1
    
    return _apply_symmetry(cell).tolist()

def _quick_gap_estimate(cell, size=32, n_pw=13, n_kpts=6):
    """Fast surrogate evaluation using reduced plane-wave basis."""
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    eps = eps_low + (eps_high - eps_low) * eps
    
    # Compute Fourier coefficients
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
    
    eps_ft = np.zeros((len(pw_idx), len(pw_idx)), dtype=complex)
    coords = np.arange(size) / size
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            phase = np.exp(-1j * 2.0 * np.pi * (m * coords + n * coords))
            phase2d = phase[:, None] * phase[None, :]
            eps_ft[im, in_] = np.sum(eps * phase2d) / (size * size)
    
    # Evaluate at key k-points
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # X
        (0.5, 0.5),      # M
        (0.25, 0.25),    # interior
        (0.3, 0.1),      # interior
        (0.1, 0.3),      # interior
    ][:n_kpts]
    
    band4_max = 0.0
    band5_min = np.inf
    
    for kx, ky in k_points:
        # Build TM eigenvalue matrix
        n_pw_actual = len(pw_idx)
        H = np.zeros((n_pw_actual * n_pw_actual, n_pw_actual * n_pw_actual), dtype=complex)
        
        for im, m in enumerate(pw_idx):
            for in_, n in enumerate(pw_idx):
                idx = im * n_pw_actual + in_
                kx_g = kx + m
                ky_g = ky + n
                k_g_sq = kx_g**2 + ky_g**2
                
                for im2, m2 in enumerate(pw_idx):
                    for in2, n2 in enumerate(pw_idx):
                        idx2 = im2 * n_pw_actual + in2
                        dm = m - m2
                        dn = n - n2
                        
                        # Find index of (dm, dn) in pw_idx
                        dm_idx = None
                        dn_idx = None
                        for p, pm in enumerate(pw_idx):
                            if pm == dm:
                                dm_idx = p
                                break
                        for p, pn in enumerate(pw_idx):
                            if pn == dn:
                                dn_idx = p
                                break
                        
                        if dm_idx is not None and dn_idx is not None:
                            H[idx, idx2] += k_g_sq * eps_ft[dm_idx, dn_idx]
                        else:
                            if dm == 0 and dn == 0:
                                H[idx, idx2] += k_g_sq * eps_ft[0, 0]
        
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.real(eigvals)
        
        # Get bands 4 and 5 (1-indexed, so indices 3 and 4)
        if len(eigvals) >= 5:
            band4_max = max(band4_max, eigvals[3])
            band5_min = min(band5_min, eigvals[4])
    
    if band5_min <= band4_max:
        return -1.0
    
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap == 0:
        return 0.0
    
    return 2.0 * gap / midgap

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    best_cell = None
    best_score = -np.inf
    
    # Strategy 1: Four-rod lattice with optimized parameters
    offsets = [0.20, 0.22, 0.25, 0.27, 0.30, 0.32, 0.35]
    radii = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    shapes = [0.0, 0.5, 1.0]
    
    for shape in shapes:
        if time.time() > deadline - 60.0:
            break
        for offset in offsets:
            if time.time() > deadline - 30.0:
                break
            for radius in radii:
                if time.time() > deadline - 10.0:
                    break
                cell = _make_four_rod_cell(size, offset, radius, shape)
                score = _quick_gap_estimate(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Strategy 2: Cross-shaped patterns
    arm_widths = [0.15, 0.20, 0.25, 0.30, 0.35]
    arm_lengths = [0.40, 0.50, 0.60, 0.70, 0.80]
    corner_radii = [0.0, 0.05, 0.10]
    
    for cr in corner_radii:
        if time.time() > deadline - 40.0:
            break
        for aw in arm_widths:
            if time.time() > deadline - 20.0:
                break
            for al in arm_lengths:
                if time.time() > deadline - 5.0:
                    break
                cell = _make_cross_cell(size, aw, al, cr)
                score = _quick_gap_estimate(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Strategy 3: Staggered rod arrays
    n_rods = [2, 3]
    rod_radii = [0.08, 0.10, 0.12, 0.14, 0.16]
    row_offsets = [0.0, 0.05, 0.10]
    
    for nr in n_rods:
        if time.time() > deadline - 30.0:
            break
        for rr in rod_radii:
            if time.time() > deadline - 15.0:
                break
            for ro in row_offsets:
                if time.time() > deadline - 5.0:
                    break
                cell = _make_staggered_cell(size, nr, rr, ro)
                score = _quick_gap_estimate(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Fine-tuning: perturb the best cell found
    if best_cell is not None and time.time() < deadline - 20.0:
        best_arr = np.array(best_cell)
        # Try flipping individual pixels near boundaries
        for i in range(size):
            if time.time() > deadline - 10.0:
                break
            for j in range(size):
                if time.time() > deadline - 5.0:
                    break
                # Only consider boundary pixels
                if best_arr[i, j] == 1:
                    # Check if it's a boundary pixel
                    is_boundary = False
                    if i > 0 and best_arr[i-1, j] == 0: is_boundary = True
                    if i < size-1 and best_arr[i+1, j] == 0: is_boundary = True
                    if j > 0 and best_arr[i, j-1] == 0: is_boundary = True
                    if j < size-1 and best_arr[i, j+1] == 0: is_boundary = True
                    
                    if is_boundary:
                        trial = best_arr.copy()
                        trial[i, j] = 0
                        trial = _apply_symmetry(trial)
                        score = _quick_gap_estimate(trial.tolist(), size)
                        if score > best_score:
                            best_score = score
                            best_cell = trial.tolist()
    
    if best_cell is None:
        # Fallback: simple four-rod
        best_cell = _make_four_rod_cell(size, 0.25, 0.12, 0.0)
    
    return best_cell
# EVOLVE-BLOCK-END
