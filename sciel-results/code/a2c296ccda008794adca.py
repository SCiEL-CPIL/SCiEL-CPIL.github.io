"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _elapsed(t0):
    return time.time() - t0

def _fourier_coefficients(cell, N):
    """Compute Fourier coefficients of 1/epsilon for a 32x32 pixel cell.
    
    For TM polarization, the eigenvalue equation involves 1/eps.
    We need the Fourier coefficients of 1/eps(k) = sum_G G_{-G} exp(i G.r)
    where G_{-G} = (1/A) sum_{r in cell} (1/eps(r)) exp(-i G.r)
    
    For a pixelated cell with eps = eps_low (air=1) or eps_high (dielectric=12),
    1/eps is 1 for air and 1/12 for dielectric.
    
    We compute G coefficients for |Gx|,|Gy| <= (N-1)/2 where N is odd (289 = 17^2, so N=17).
    """
    size = len(cell)
    # N = 17, so we need Gx, Gy from -8 to 8
    Npw = 17  # 289 = 17^2 plane waves
    half = (Npw - 1) // 2  # 8
    
    # Precompute pixel positions
    # Pixel (i,j) center is at ((i+0.5)/size - 0.5, (j+0.5)/size - 0.5)
    # In units of lattice constant a=1
    
    # 1/eps values
    eps_inv = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                eps_inv[i][j] = 1.0 / 12.0
            else:
                eps_inv[i][j] = 1.0
    
    # Fourier coefficients: G_{-G} = (1/A) sum_{pixels} (1/eps(r)) * exp(-i G.r) * pixel_area
    # pixel_area = 1/size^2
    # G = (2*pi*nx, 2*pi*ny) for integer nx, ny
    # G.r = 2*pi*(nx*x + ny*y)
    
    coeff = np.zeros((2*half+1, 2*half+1), dtype=complex)
    
    for nx in range(-half, half+1):
        for ny in range(-half, half+1):
            s = 0.0 + 0.0j
            for i in range(size):
                x = (i + 0.5) / size - 0.5
                for j in range(size):
                    y = (j + 0.5) / size - 0.5
                    phase = math.exp(-2j * math.pi * (nx * x + ny * y))
                    s += eps_inv[i][j] * phase
            coeff[nx + half, ny + half] = s / (size * size)
    
    return coeff, Npw, half

def _build_eigenmatrix(coeff, Npw, half, kx, ky):
    """Build the eigenvalue matrix for the plane-wave method.
    
    For TM: det[(k+G)^2 - (omega/c)^2 * eps_inv_{G-G'}] = 0
    This becomes a generalized eigenvalue problem, but since we're using
    the 1/eps formulation, it's actually:
    
    sum_{G'} (k+G)^2 delta_{G,G'} * c_{G'} = (omega/c)^2 * sum_{G'} eps_inv_{G-G'} c_{G'}
    
    Wait, let me reconsider. The standard plane-wave method for photonic crystals:
    
    For TM (Ez nonzero):
    (k+G)^2 * A_{G} = (omega/c)^2 * sum_{G'} (1/eps)_{G-G'} * A_{G'}
    
    So we need to solve: M * A = lambda * A where M_{G,G'} = (k+G)^2 * delta_{G,G'} / (1/eps)_{G-G'}
    
    Actually, the standard formulation is:
    sum_{G'} (1/eps)_{G-G'} (k+G')^2 A_{G'} = (omega/c)^2 A_{G}
    
    This is a standard eigenvalue problem: M A = omega^2 A
    where M_{G,G'} = (1/eps)_{G-G'} (k+G')^2
    
    Wait no. Let me be more careful.
    
    Maxwell's equation for TM: ∇²Ez + (omega/c)^2 eps(r) Ez = 0
    With Ez = sum_G A_G exp(i(k+G).r)
    eps(r) Ez = sum_G' eps_{G-G'} sum_G A_G exp(i(k+G).r)
    
    The equation is: (k+G)^2 A_G = (omega/c)^2 sum_{G'} (1/eps)_{G-G'} A_{G'}
    
    So: sum_{G'} (1/eps)_{G-G'} A_{G'} - (c/omega)^2 (k+G)^2 A_G = 0
    
    Or equivalently: sum_{G'} [(1/eps)_{G-G'} / (k+G)^2] A_{G'} = (c/omega)^2 A_G
    
    Hmm, this doesn't quite work because (k+G)^2 can be zero or negative.
    
    The correct formulation: The eigenvalue equation is
    (k+G)^2 A_G = (omega/c)^2 sum_{G'} eps_inv_{G-G'} A_{G'}
    
    This is a generalized eigenvalue problem: D A = (omega/c)^2 V A
    where D is diagonal with D_{G,G} = (k+G)^2 and V_{G,G'} = eps_inv_{G-G'}
    
    To convert to standard form, we can write: V^{-1} D A = (omega/c)^2 A
    But V may not be easily invertible.
    
    Alternative: multiply both sides by 1/(k+G)^2 (assuming nonzero):
    A_G = (omega/c)^2 sum_{G'} eps_inv_{G-G'} A_{G'} / (k+G)^2
    
    This gives: sum_{G'} [eps_inv_{G-G'} / (k+G)^2] A_{G'} = (omega/c)^2 A_G
    
    But this has issues when (k+G)^2 = 0.
    
    The most robust approach: use the generalized eigenvalue solver or reformulate.
    
    Actually, the standard approach in the literature (Joannopoulos et al.) is:
    
    For TM: The eigenvalue equation is
    sum_{G'} (1/eps)_{G-G'} (k+G')^2 A_{G'} = (omega/c)^2 A_{G}
    
    Wait, I think I had the indices wrong. Let me redo:
    
    ∇²Ez = -sum_G (k+G)^2 A_G exp(i(k+G).r)
    eps(r) Ez = sum_{G,G'} eps_{G'-G} A_G exp(i(k+G').r)  [relabeling]
    
    So the equation -∇²Ez = (omega/c)^2 eps Ez becomes:
    sum_G (k+G)^2 A_G exp(i(k+G).r) = (omega/c)^2 sum_{G,G'} eps_{G'-G} A_G exp(i(k+G').r)
    
    Setting G' = G + G'' where G'' is a reciprocal lattice vector:
    (k+G)^2 A_G = (omega/c)^2 sum_{G''} eps_{G''} A_{G-G''}
    
    Let G' = G - G'':
    (k+G)^2 A_G = (omega/c)^2 sum_{G'} eps_{G-G'} A_{G'}
    
    So: sum_{G'} eps_{G-G'} A_{G'} - (c/omega)^2 (k+G)^2 A_G = 0
    
    This is: V A = (c/omega)^2 D A  where V_{G,G'} = eps_{G-G'}, D_{G,G} = (k+G)^2
    
    Or: V^{-1} D A = (c/omega)^2 A, i.e., omega^2 = c^2 * eigenvalue of V^{-1} D
    
    Hmm, but we want to use 1/eps. Let me reconsider.
    
    Actually, for the "exact coefficient" method with pixelated eps, the standard approach is:
    
    The eigenvalue equation for TM polarization is:
    (k+G)^2 A_G = (omega/c)^2 sum_{G'} (1/eps)_{G-G'}^{-1} ... 
    
    No wait. Let me look at this differently.
    
    The key equation is: ∇ × ∇ × E = (omega/c)^2 eps E
    For TM (E = Ez zhat): ∇ × ∇ × (Ez zhat) = -∇²Ez zhat + ∇(∂Ez/∂z) zhat = -∇²Ez zhat (for 2D, ∂/∂z=0)
    
    So: -∇²Ez = (omega/c)^2 eps(r) Ez
    
    With Ez = sum_G A_G e^{i(k+G).r}:
    sum_G (k+G)^2 A_G e^{i(k+G).r} = (omega/c)^2 sum_{G,G'} eps_{G'-G} A_G e^{i(k+G').r}
    
    Renaming: let the RHS index be G' (the output plane wave), and G is the input:
    (k+G)^2 A_G = (omega/c)^2 sum_{G'} eps_{G-G'} A_{G'}
    
    where eps_{G-G'} = Fourier coefficient of eps at reciprocal vector G-G'.
    
    So the eigenvalue problem is:
    D A = (omega/c)^2 V A  where D_{G,G}=(k+G)^2, V_{G,G'}=eps_{G-G'}
    
    This is a generalized eigenvalue problem. We can solve it as:
    V^{-1} D A = (omega/c)^2 A
    
    Since V is Hermitian positive definite (eps > 0), V^{-1} exists.
    
    Alternatively, we can use the 1/eps formulation which is more standard:
    Multiply both sides by 1/eps... no, that doesn't simplify.
    
    Let me just use numpy's eigensolver for the generalized problem, or better,
    convert to standard form.
    
    Actually, the most common formulation in photonic crystal literature uses:
    For TM: M_{G,G'} = (1/eps)_{G-G'} and the equation is
    sum_{G'} M_{G,G'} (k+G')^2 A_{G'} = (omega/c)^2 A_{G}
    
    Wait, I think there are two equivalent formulations. Let me use the one that's
    most numerically stable.
    
    Formulation 1: D A = omega^2 V A (generalized)
    Formulation 2: V^{-1} D A = omega^2 A (standard, but need to invert V)
    Formulation 3: Using 1/eps: sum_{G'} (1/eps)_{G-G'} A_{G'} = (c/omega)^2 (k+G)^2 A_G
    This gives: (1/eps) matrix * A = (c/omega)^2 D * A, which is again generalized.
    
    I'll use the direct approach: solve the generalized eigenvalue problem
    D A = lambda V A using scipy.linalg.eig or by converting to standard form.
    
    Actually, for efficiency, let me use the standard formulation:
    M A = omega^2 A where M = V^{-1} D
    
    But computing V^{-1} for 289x289 is feasible.
    
    Alternatively, I can use scipy.linalg.eigh for the generalized problem if both
    matrices are Hermitian.
    
    Let me use: scipy.linalg.eigh(D, V) to solve D A = omega^2 V A
    """
    pass

def _estimate_band_gap(cell, Npw=17):
    """Estimate the TM band gap between bands 4 and 5 for a given cell.
    
    Uses plane-wave method with exact Fourier coefficients.
    Returns the gap-midgap ratio, or a negative value if bands overlap.
    """
    t0 = time.time()
    
    size = len(cell)
    half = (Npw - 1) // 2
    
    # Compute 1/eps Fourier coefficients
    # For TM, we need eps Fourier coefficients
    # eps(r) = eps_low for air, eps_high for dielectric
    # eps_G = (1/A) sum_r eps(r) exp(-i G.r)
    
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute eps Fourier coefficients
    eps_coeff = np.zeros((2*half+1, 2*half+1), dtype=complex)
    for nx in range(-half, half+1):
        for ny in range(-half, half+1):
            s = 0.0 + 0.0j
            for i in range(size):
                x = (i + 0.5) / size - 0.5
                for j in range(size):
                    y = (j + 0.5) / size - 0.5
                    if cell[i][j] == 1:
                        eps_val = eps_high
                    else:
                        eps_val = eps_low
                    phase = math.exp(-2j * math.pi * (nx * x + ny * y))
                    s += eps_val * phase
            eps_coeff[nx + half, ny + half] = s / (size * size)
    
    # Build list of plane waves
    G_list = []
    for nx in range(-half, half+1):
        for ny in range(-half, half+1):
            G_list.append((2.0 * math.pi * nx, 2.0 * math.pi * ny))
    N = len(G_list)  # should be 289
    
    # Irreducible BZ wave vectors (30 points)
    # For square lattice, irreducible BZ is the triangle with vertices
    # (0,0), (pi, 0), (pi, pi) in units of 1/a
    # Let's sample 30 points in this triangle
    
    k_points = []
    # Along Gamma-X: (0,0) to (pi, 0)
    for t in np.linspace(0, 1, 10):
        k_points.append((math.pi * t, 0.0))
    # Along X-M: (pi, 0) to (pi, pi)
    for t in np.linspace(0, 1, 10):
        k_points.append((math.pi, math.pi * t))
    # Along M-Gamma: (pi, pi) to (0, 0)
    for t in np.linspace(0, 1, 10):
        k_points.append((math.pi * (1-t), math.pi * (1-t)))
    
    # Remove duplicates
    unique_k = []
    for k in k_points:
        found = False
        for uk in unique_k:
            if abs(k[0]-uk[0]) < 1e-6 and abs(k[1]-uk[1]) < 1e-6:
                found = True
                break
        if not found:
            unique_k.append(k)
    k_points = unique_k[:30]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        # Build D matrix: D_{G,G} = (kx+Gx)^2 + (ky+Gy)^2
        D = np.zeros((N, N))
        for idx, (Gx, Gy) in enumerate(G_list):
            D[idx, idx] = (kx + Gx)**2 + (ky + Gy)**2
        
        # Build V matrix: V_{G,G'} = eps_{G-G'}
        # eps_{G-G'} = eps_coeff at index corresponding to (Gx-G'x, Gy-G'y)
        V = np.zeros((N, N), dtype=complex)
        for idx, (Gx, Gy) in enumerate(G_list):
            for jdx, (Gpx, Gpy) in enumerate(G_list):
                dnx = int(round((Gx - Gpx) / (2.0 * math.pi)))
                dny = int(round((Gy - Gpy) / (2.0 * math.pi)))
                if -half <= dnx <= half and -half <= dny <= half:
                    V[idx, jdx] = eps_coeff[dnx + half, dny + half]
        
        # Solve generalized eigenvalue problem: D A = omega^2 V A
        # Use scipy
        try:
            from scipy.linalg import eigh
            # Both D and V should be Hermitian
            D = (D + D.T.conj()) / 2
            V = (V + V.T.conj()) / 2
            eigenvalues, _ = eigh(D, V, subset_by_index=[3, 4])
            # eigenvalues are omega^2/c^2, so omega = sqrt(eigenvalue) * c
            # We work in units where c=1, so omega = sqrt(eigenvalue)
            omega4 = math.sqrt(max(eigenvalues[0], 0))
            omega5 = math.sqrt(max(eigenvalues[1], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        except Exception:
            continue
        
        if _elapsed(t0) > 100:
            break
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap == 0:
        return -1.0
    return 2.0 * gap / midgap

def _make_cell_symmetric(cell):
    """Ensure the cell has full square symmetry (D4): 180° rotation, mirror x, mirror y, transpose."""
    size = len(cell)
    # Apply all symmetry operations and average, then threshold
    result = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # Original
            result[i, j] += cell[i][j]
            # 180 rotation
            result[size-1-i, size-1-j] += cell[i][j]
            # Mirror x (left-right)
            result[i, size-1-j] += cell[i][j]
            # Mirror y (top-bottom)
            result[size-1-i, j] += cell[i][j]
            # Transpose
            result[j, i] += cell[i][j]
    
    # Threshold at 2.5 (out of 5 operations, need at least 3 to be dielectric)
    sym_cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if result[i, j] >= 3:
                sym_cell[i, j] = 1
    
    return sym_cell.tolist()

def _make_ring_cell(size, inner_r, outer_r, thickness_profile=None):
    """Create a symmetric ring-shaped dielectric cell.
    
    The ring is centered at the cell center.
    """
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            r = math.sqrt(dx*dx + dy*dy)
            if inner_r <= r <= outer_r:
                cell[i, j] = 1
    return cell.tolist()

def _make_cross_cell(size, arm_width):
    """Create a symmetric cross-shaped dielectric cell."""
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    half_w = arm_width / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - center) <= half_w or abs(j - center) <= half_w:
                cell[i, j] = 1
    return cell.tolist()

def _make_diamond_cell(size, radius):
    """Create a symmetric diamond (rotated square) dielectric cell."""
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            if (dx + dy) / math.sqrt(2) <= radius:
                cell[i, j] = 1
    return cell.tolist()

def _make_double_ring_cell(size, r1_in, r1_out, r2_in, r2_out):
    """Create a cell with two concentric dielectric rings."""
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            r = math.sqrt(dx*dx + dy*dy)
            if (r1_in <= r <= r1_out) or (r2_in <= r <= r2_out):
                cell[i, j] = 1
    return cell.tolist()

def _make_hexagonal_cell(size, radius):
    """Create a symmetric hexagonal dielectric cell (approximated on square grid)."""
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            # Hexagon: max(|dx|, |dy|, |dx+dy|/sqrt(2), |dx-dy|/sqrt(2)) <= radius
            # Simpler: use the hexagonal distance
            r_hex = max(abs(dx), abs(dy), abs(dx + dy) / math.sqrt(2), abs(dx - dy) / math.sqrt(2))
            if r_hex <= radius:
                cell[i, j] = 1
    return cell.tolist()

def _make_pattern_cell(size, pattern_type, params):
    """Generate various symmetric patterns."""
    if pattern_type == 'rod':
        r = params.get('radius', 0.2)
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = (i + 0.5) / size - 0.5
                dy = (j + 0.5) / size - 0.5
                if dx*dx + dy*dy < r*r:
                    cell[i, j] = 1
        return cell.tolist()
    elif pattern_type == 'ring':
        return _make_ring_cell(size, params.get('inner_r', 5), params.get('outer_r', 10))
    elif pattern_type == 'cross':
        return _make_cross_cell(size, params.get('arm_width', 6))
    elif pattern_type == 'diamond':
        return _make_diamond_cell(size, params.get('radius', 10))
    elif pattern_type == 'double_ring':
        return _make_double_ring_cell(size, 
                                       params.get('r1_in', 4), params.get('r1_out', 7),
                                       params.get('r2_in', 10), params.get('r2_out', 13))
    elif pattern_type == 'hexagon':
        return _make_hexagonal_cell(size, params.get('radius', 10))
    elif pattern_type == 'grid':
        # Grid of small rods
        spacing = params.get('spacing', 8)
        rod_r = params.get('rod_r', 2)
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                # Check if near any grid point
                for gi in range(-size//spacing, size//spacing + 1):
                    for gj in range(-size//spacing, size//spacing + 1):
                        cx = center + gi * spacing
                        cy = center + gj * spacing
                        if (i-cx)**2 + (j-cy)**2 < rod_r*rod_r:
                            cell[i, j] = 1
        return cell.tolist()
    elif pattern_type == 'frame':
        # Frame: dielectric border with air interior
        border = params.get('border', 3)
        cell = np.ones((size, size), dtype=int)
        cell[border:size-border, border:size-border] = 0
        return cell.tolist()
    elif pattern_type == 'checker':
        # Checkerboard pattern with period
        period = params.get('period', 4)
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                # Symmetric checkerboard
                ii = int(abs(i - center))
                jj = int(abs(j - center))
                if (ii // period + jj // period) % 2 == 0:
                    cell[i, j] = 1
        return cell.tolist()
    else:
        # Default: rod
        return _make_ring_cell(size, 0, 6)

def _local_optimize(cell, max_steps=200, time_limit=100):
    """Local optimization: flip pixels to improve the gap estimate.
    
    Uses a simple hill-climbing approach.
    """
    t0 = time.time()
    size = len(cell)
    best_cell = [row[:] for row in cell]
    best_score = _estimate_band_gap(best_cell)
    
    if best_score < 0:
        best_score = -1.0
    
    step = 0
    while step < max_steps and _elapsed(t0) < time_limit:
        step += 1
        improved = False
        
        # Try flipping each pixel (in random order for diversity)
        import random
        indices = list(range(size * size))
        random.shuffle(indices)
        
        for idx in indices:
            if _elapsed(t0) > time_limit - 5:
                break
            
            i = idx // size
            j = idx % size
            
            # Try flipping
            old_val = best_cell[i][j]
            best_cell[i][j] = 1 - old_val
            
            # Ensure symmetry: apply all D4 operations
            sym_cell = _make_cell_symmetric(best_cell)
            
            score = _estimate_band_gap(sym_cell)
            if score > best_score + 1e-6:
                best_score = score
                best_cell = sym_cell
                improved = True
                # Continue searching from new position
            else:
                best_cell[i][j] = old_val
        
        if not improved and step > 10:
            # Try some random perturbations
            for _ in range(5):
                if _elapsed(t0) > time_limit - 5:
                    break
                # Flip a random cluster
                ci = random.randint(2, size-3)
                cj = random.randint(2, size-3)
                radius = random.randint(1, 3)
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        ni, nj = ci+di, cj+dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if di*di + dj*dj <= radius*radius:
                                best_cell[ni][nj] = 1 - best_cell[ni][nj]
                
                sym_cell = _make_cell_symmetric(best_cell)
                score = _estimate_band_gap(sym_cell)
                if score > best_score + 1e-6:
                    best_score = score
                    best_cell = sym_cell
                    improved = True
    
    return best_cell, best_score

def design(problem):
    """Design a symmetric 32x32-pixel unit cell for a photonic crystal
    that maximizes the TM band gap between bands 4 and 5."""
    t0 = time.time()
    WALL_LIMIT = 220  # Safety margin before 230s kill
    
    size = problem.get("pixels", 32)
    
    # Generate initial candidates from various pattern types
    candidates = []
    
    # 1. Textbook rod (baseline)
    candidates.append(('rod', {'radius': 0.2}))
    
    # 2. Various rod sizes
    for r in [0.1, 0.15, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(('rod', {'radius': r}))
    
    # 3. Rings
    for inner in [3, 5, 7, 9]:
        for outer in [inner+2, inner+3, inner+4, inner+5]:
            if outer < size/2:
                candidates.append(('ring', {'inner_r': inner, 'outer_r': outer}))
    
    # 4. Crosses
    for w in [4, 6, 8, 10, 12]:
        candidates.append(('cross', {'arm_width': w}))
    
    # 5. Diamonds
    for r in [6, 8, 10, 12]:
        candidates.append(('diamond', {'radius': r}))
    
    # 6. Double rings
    for r1i, r1o in [(3,6), (5,8), (4,7)]:
        for r2i, r2o in [(10,13), (11,14), (9,12)]:
            if r2o < size/2:
                candidates.append(('double_ring', {
                    'r1_in': r1i, 'r1_out': r1o,
                    'r2_in': r2i, 'r2_out': r2o
                }))
    
    # 7. Hexagons
    for r in [6, 8, 10, 12]:
        candidates.append(('hexagon', {'radius': r}))
    
    # 8. Grids
    for spacing in [6, 8, 10, 12]:
        for rod_r in [1, 2, 3]:
            candidates.append(('grid', {'spacing': spacing, 'rod_r': rod_r}))
    
    # 9. Frames
    for border in [2, 3, 4, 5, 6]:
        candidates.append(('frame', {'border': border}))
    
    # 10. Checkerboards
    for period in [3, 4, 5, 6, 8]:
        candidates.append(('checker', {'period': period}))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    
    for ptype, params in candidates:
        if time.time() - t0 > WALL_LIMIT - 60:
            break
        
        cell = _make_pattern_cell(size, ptype, params)
        cell = _make_cell_symmetric(cell)
        
        score = _estimate_band_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    if best_cell is None:
        # Fallback to the original rod
        best_cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                row.append(1 if x*x + y*y < 0.2**2 else 0)
            best_cell.append(row)
        best_score = _estimate_band_gap(best_cell)
    
    # Local optimization if time permits
    remaining = WALL_LIMIT - 20 - (time.time() - t0)
    if remaining > 30:
        opt_cell, opt_score = _local_optimize(best_cell, max_steps=150, time_limit=min(remaining, 120))
        if opt_score > best_score:
            best_cell = opt_cell
            best_score = opt_score
    
    # Final safety check
    if time.time() - t0 > WALL_LIMIT - 10:
        # Return immediately
        pass
    
    # Ensure the result is a proper nested list of 0/1
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 if best_cell[i][j] else 0)
        final_cell.append(row)
    
    return final_cell
# EVOLVE-BLOCK-END
