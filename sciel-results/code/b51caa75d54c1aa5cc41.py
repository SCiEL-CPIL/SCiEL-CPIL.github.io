import time
import numpy as np
import math

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0

# Plane-wave basis: 289 = 17x17 grid of G vectors
_G_RANGE = 8
_GV = [(m, n) for m in range(-_G_RANGE, _G_RANGE+1) for n in range(-_G_RANGE, _G_RANGE+1)]
_NPW = len(_GV)  # 289

# Precompute G index lookup
_G_INDEX = {}
for idx, (gm, gn) in enumerate(_GV):
    _G_INDEX[(gm, gn)] = idx

# Precompute difference indices for M matrix construction
# M[i,j] needs mu[G[i] - G[j]]
_DIFF_IDX = np.zeros((_NPW, _NPW), dtype=np.int32)
for i in range(_NPW):
    gi, gj_ = _GV[i]
    for j in range(_NPW):
        gj, gk = _GV[j]
        dm = gi - gj
        dn = gj_ - gk
        if -_G_RANGE <= dm <= _G_RANGE and -_G_RANGE <= dn <= _G_RANGE:
            _DIFF_IDX[i, j] = _G_INDEX[(dm, dn)]
        else:
            _DIFF_IDX[i, j] = -1

# K-points: 30 points in the irreducible BZ
_KPOINTS = []
# Triangle vertices: (0,0), (0.25, 0.25), (0, 0.5)
# We sample a 6x6 grid in (u,v) where k = u*e1 + v*e2, e1=(0.25,0.25), e2=(0,0.5)
# Constraint: u+v <= 1
for i in range(6):
    for j in range(6):
        u = i / 5.0
        v = j / 5.0
        if u + v <= 1.0:
            kx = u * 0.25
            ky = u * 0.25 + v * 0.5
            _KPOINTS.append((kx, ky))

# Deduplicate and limit to 30
_seen = set()
_unique_kpts = []
for p in _KPOINTS:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in _seen:
        _seen.add(key)
        _unique_kpts.append(p)
_KPOINTS = _unique_kpts[:30]

# Precompute (k+G)^2 for all k-points and G-vectors
# shape: (N_K, N_PW)
_KG_SQ = np.zeros((len(_KPOINTS), _NPW), dtype=np.float64)
_KG_MAG = np.zeros((len(_KPOINTS), _NPW), dtype=np.float64)
for ki, (kx, ky) in enumerate(_KPOINTS):
    for i in range(_NPW):
        gm, gn = _GV[i]
        kx_tot = kx + gm
        ky_tot = ky + gn
        _KG_SQ[ki, i] = kx_tot * kx_tot + ky_tot * ky_tot
        _KG_MAG[ki, i] = math.sqrt(_KG_SQ[ki, i])

# Precompute M matrix template indices
# M is determined by coeffs. We can precompute the structure of M.
# But coeffs change. So we just use _DIFF_IDX.

def _compute_fourier_coeffs(cell_flat):
    """Compute Fourier coefficients of 1/eps for a given cell (flat array)."""
    # cell_flat is 0/1. 1 is dielectric.
    eps = np.where(cell_flat == 1, _EPS_HIGH, _EPS_LOW)
    eps_inv = 1.0 / eps
    # Reshape to 2D for FFT
    eps_inv_2d = eps_inv.reshape(_SIZE, _SIZE)
    # FFT
    fft_coeffs = np.fft.fft2(eps_inv_2d) / (_SIZE * _SIZE)
    
    coeffs = np.zeros(_NPW, dtype=np.complex128)
    for idx, (gm, gn) in enumerate(_GV):
        idx_m = gm % _SIZE
        idx_n = gn % _SIZE
        coeffs[idx] = fft_coeffs[idx_m, idx_n]
    return coeffs

def _tm_bands_fast(cell):
    """Compute TM band-4/5 gap ratio using vectorized NumPy solver."""
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    coeffs = _compute_fourier_coeffs(cell_flat)
    
    # Construct M matrix (common for all k-points)
    # M[i,j] = coeffs[DIFF_IDX[i,j]]
    M = np.zeros((_NPW, _NPW), dtype=np.complex128)
    valid = _DIFF_IDX >= 0
    M[valid] = coeffs[_DIFF_IDX[valid]]
    
    # H_k = M * outer(kg_mag[k], kg_mag[k])
    # We can compute eigenvalues for all k-points in a loop, but optimized.
    
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for ki in range(len(_KPOINTS)):
        # H = M * (kg_mag[ki] * kg_mag[ki].T)
        # This is a rank-1 update of M, but M is not diagonal.
        # Actually H_{ij} = M_{ij} * kg_i * kg_j
        # We can compute H = M * np.outer(_KG_MAG[ki], _KG_MAG[ki])
        
        # To avoid allocating huge arrays repeatedly, we can use einsum or just multiply
        # H = M * np.outer(_KG_MAG[ki], _KG_MAG[ki])
        
        # Optimization: Precompute outer products? No, _KG_MAG changes per k.
        # But _KG_MAG is precomputed.
        
        # H = M * np.outer(_KG_MAG[ki], _KG_MAG[ki])
        # This creates a 289x289 array.
        
        # Alternative: Solve generalized eigenvalue problem? No, H is Hermitian.
        
        # Let's just do it. 30 iterations of 289x289 eigvalsh is fast in NumPy.
        
        H = M * np.outer(_KG_MAG[ki], _KG_MAG[ki])
        H = 0.5 * (H + H.conj().T) # Ensure Hermitian
        
        eigvals = np.linalg.eigvalsh(H)
        
        if len(eigvals) >= 5:
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            if w4_sq > 0 and w5_sq > 0:
                w4 = math.sqrt(w4_sq)
                w5 = math.sqrt(w5_sq)
                band_4_max = max(band_4_max, w4)
                band_5_min = min(band_5_min, w5)
    
    if band_4_max <= 0 or band_5_min <= 0:
        return -1.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / (band_5_min + band_4_max)

def _generate_cell_from_params(params):
    """
    Generate a 32x32 cell from parameters.
    params: [r, e1, e2, cx, cy]
    r: radius scale
    e1, e2: exponents for superellipse (x^e1 + y^e2 < r^e)
    cx, cy: offset from center (0,0) for one quadrant, mirrored.
    """
    r = params[0]
    e1 = params[1]
    e2 = params[2]
    cx = params[3]
    cy = params[4]
    
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    
    # Centers in normalized coordinates [-0.5, 0.5]
    # We place 4 centers at (cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy)
    # Note: cx, cy are offsets from 0.25? No, let's use absolute coordinates.
    # Let's define centers relative to the quarter points (0.25, 0.25) etc.
    # Better: define centers at (0.25+cx, 0.25+cy) etc.
    # But symmetry requires cell == cell[::-1] and transpose.
    # If we place a shape at (0.25+cx, 0.25+cy), we must also place it at:
    # (0.25-cx, 0.25+cy), (0.25+cx, 0.25-cy), (0.25-cx, 0.25-cy)
    # This preserves C4v symmetry about the center (0,0) if the base is centered at 0.25.
    
    centers = [
        (0.25 + cx, 0.25 + cy),
        (0.25 - cx, 0.25 + cy),
        (0.25 + cx, 0.25 - cy),
        (0.25 - cx, 0.25 - cy)
    ]
    
    # Precompute pixel coordinates
    xs = np.arange(_SIZE) / _SIZE - 0.5 + 0.5 / _SIZE
    ys = np.arange(_SIZE) / _SIZE - 0.5 + 0.5 / _SIZE
    X, Y = np.meshgrid(xs, ys)
    
    for cx_c, cy_c in centers:
        # Superellipse: |x|^e1 + |y|^e2 <= r^e
        # To handle exponents, we use (x^2)^e1/2 + (y^2)^e2/2
        dx = X - cx_c
        dy = Y - cy_c
        
        # Ensure positive bases for power
        term_x = np.power(np.abs(dx), e1)
        term_y = np.power(np.abs(dy), e2)
        
        mask = (term_x + term_y) <= np.power(r, e1) # Approximation for superellipse
        # More accurate superellipse: (|dx|/r)^e1 + (|dy|/r)^e2 <= 1
        # Let's use the standard form:
        # (|dx|/r)^e + (|dy|/r)^e <= 1 for superellipse with single exponent e.
        # But we have e1, e2. Let's stick to single exponent e for simplicity and speed.
        # params: [r, e, cx, cy]
        # Let's re-define params in the caller to match this.
        
    # The above loop is inefficient. Let's vectorize the whole thing.
    # We'll rewrite _generate_cell_from_params to be fully vectorized.
    
    return cell

def _generate_cell_fast(params):
    """
    Vectorized cell generation.
    params: [r, e, cx, cy]
    r: radius
    e: exponent (2=circle, 4=square-ish, higher=boxier)
    cx, cy: offset from quarter-center (0.25, 0.25)
    """
    r = params[0]
    e = params[1]
    cx = params[2]
    cy = params[3]
    
    # Pixel coordinates
    # i, j in 0..31
    # x = (i + 0.5)/32 - 0.5
    # y = (j + 0.5)/32 - 0.5
    idx = np.arange(_SIZE)
    xs = (idx + 0.5) / _SIZE - 0.5
    ys = (idx + 0.5) / _SIZE - 0.5
    X, Y = np.meshgrid(xs, ys)
    
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    
    # Centers: (0.25+cx, 0.25+cy) and its symmetric counterparts
    # We check if pixel is inside ANY of the 4 shapes.
    
    # Shape 1: center (0.25+cx, 0.25+cy)
    # Shape 2: center (0.25-cx, 0.25+cy)
    # Shape 3: center (0.25+cx, 0.25-cy)
    # Shape 4: center (0.25-cx, 0.25-cy)
    
    # To be efficient, we can check the condition for one and mirror?
    # No, just check all 4.
    
    # Superellipse condition: (|dx|/r)^e + (|dy|/r)^e <= 1
    # If r == 0, skip.
    
    if r <= 0:
        return cell
        
    # Precompute powers of 1/r
    inv_r = 1.0 / r
    
    # Center 1
    dx1 = np.abs(X - (0.25 + cx)) * inv_r
    dy1 = np.abs(Y - (0.25 + cy)) * inv_r
    mask1 = (np.power(dx1, e) + np.power(dy1, e)) <= 1.0
    
    # Center 2
    dx2 = np.abs(X - (0.25 - cx)) * inv_r
    dy2 = np.abs(Y - (0.25 + cy)) * inv_r
    mask2 = (np.power(dx2, e) + np.power(dy2, e)) <= 1.0
    
    # Center 3
    dx3 = np.abs(X - (0.25 + cx)) * inv_r
    dy3 = np.abs(Y - (0.25 - cy)) * inv_r
    mask3 = (np.power(dx3, e) + np.power(dy3, e)) <= 1.0
    
    # Center 4
    dx4 = np.abs(X - (0.25 - cx)) * inv_r
    dy4 = np.abs(Y - (0.25 - cy)) * inv_r
    mask4 = (np.power(dx4, e) + np.power(dy4, e)) <= 1.0
    
    final_mask = mask1 | mask2 | mask3 | mask4
    cell = final_mask.astype(np.int8)
    
    return cell

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 190.0 # Safety margin for 200s limit
    
    size = problem.get("pixels", 32)
    if size != _SIZE:
        cell = [[0]*size for _ in range(size)]
        return cell

    # Initial parameters: 4 circles at quarter points
    # r ~ 0.15, e=2 (circle), cx=0, cy=0
    best_params = np.array([0.15, 2.0, 0.0, 0.0])
    best_cell = _generate_cell_fast(best_params)
    best_score = _tm_bands_fast(best_cell)
    
    # Local search on parameters
    # Parameters: r in [0.05, 0.3], e in [1.5, 4.0], cx in [-0.1, 0.1], cy in [-0.1, 0.1]
    
    # Define bounds
    bounds = [
        (0.05, 0.3),  # r
        (1.5, 4.0),   # e
        (-0.1, 0.1),  # cx
        (-0.1, 0.1)   # cy
    ]
    
    # Simple coordinate descent with random restarts
    step_sizes = [0.02, 0.2, 0.01, 0.01]
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10.0:
        # Pick a parameter to perturb
        p_idx = np.random.randint(4)
        
        # Random step
        step = np.random.uniform(-step_sizes[p_idx], step_sizes[p_idx])
        
        new_params = best_params.copy()
        new_params[p_idx] += step
        
        # Clip to bounds
        for i in range(4):
            new_params[i] = np.clip(new_params[i], bounds[i][0], bounds[i][1])
            
        new_cell = _generate_cell_fast(new_params)
        score = _tm_bands_fast(new_cell)
        
        if score > best_score:
            best_score = score
            best_params = new_params
            best_cell = new_cell
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Random restart within bounds
            best_params = np.array([
                np.random.uniform(*bounds[0]),
                np.random.uniform(*bounds[1]),
                np.random.uniform(*bounds[2]),
                np.random.uniform(*bounds[3])
            ])
            best_cell = _generate_cell_fast(best_params)
            best_score = _tm_bands_fast(best_cell)
            no_improve_count = 0

    return best_cell.tolist()
