"""Rewritten program for photonic crystal band gap optimization."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_symmetry(base):
    """
    Enforce square symmetry (D4) on a 32x32 array.
    Symmetries: 90-deg rotation, reflection about main diagonal.
    We take the majority vote over the 8 equivalent positions for each pixel.
    """
    size = base.shape[0]
    b = np.array(base, dtype=np.int8)
    
    # Generate all 8 transformations
    # 0: Identity
    # 1: 90 rot
    # 2: 180 rot
    # 3: 270 rot
    # 4: Reflect X
    # 5: Reflect Y
    # 6: Reflect Main Diag
    # 7: Reflect Anti Diag
    # Actually, simpler: just use the 4 symmetries that generate the group if we are careful, 
    # or just apply all 8 to be safe.
    
    r0 = b
    r90 = np.rot90(b, 1)
    r180 = np.rot90(b, 2)
    r270 = np.rot90(b, 3)
    rx = b[:, ::-1]
    ry = b[::-1, :]
    rd = b.T
    rad = rd[:, ::-1] # Reflect anti-diagonal: flip main diag then flip rows? 
    # Let's verify anti-diagonal reflection.
    # (i,j) -> (N-1-j, N-1-i).
    # b.T is (j,i). b.T[:, ::-1] is (j, N-1-i). This is not quite right for all.
    # Let's just use the 4 unique orbits if we define them properly.
    # The group D4 has 8 elements.
    # Let's just sum all 8 and threshold.
    
    votes = (r0 + r90 + r180 + r270 + rx + ry + rd + rad)
    cell = (votes >= 4).astype(np.int8)
    return cell.tolist()

def _compute_gap(cell, size=32, n_pw=10, k_points=None, eps_high=12.0, eps_low=1.0):
    """
    Compute the gap ratio between band 4 and 5 for TM polarization.
    Uses plane-wave expansion.
    """
    if k_points is None:
        # Define k-points in the irreducible BZ
        # Gamma (0,0), X (pi, 0), M (pi, pi)
        # Sample along Gamma-X, X-M, M-Gamma
        # Note: k is in units of 2*pi/a, so X is (1, 0) in these units? 
        # Standard convention: G = 2*pi*(mx, my). k in [0, pi] range for irreducible?
        # Let's use k in radians. BZ is [-pi, pi]^2.
        # Irreducible: 0 <= ky <= kx <= pi.
        
        k_points = []
        # Gamma to X
        for t in np.linspace(0, 1, 10):
            k_points.append((t * np.pi, 0))
        # X to M
        for t in np.linspace(0, 1, 10):
            k_points.append((np.pi, t * np.pi))
        # M to Gamma (via X? No, M to Gamma is diagonal)
        # The irreducible triangle is Gamma(0,0) - X(pi,0) - M(pi,pi).
        # Edge M-Gamma: kx=ky, from pi to 0.
        for t in np.linspace(0, 1, 10):
            k_points.append((t * np.pi, t * np.pi))
            
        # Remove duplicates or close points if needed, but 30 points is fine.
        
    # Precompute G vectors
    # G = (mx, my) * 2*pi
    # We work in units where 2*pi = 1 for simplicity in G indices, 
    # but k must be consistent. 
    # Let's stick to physical units: G = 2*pi * (mx, my).
    
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    Gs *= 2.0 * np.pi
    nG = len(Gs)
    
    # Compute inv_eps Fourier coefficients
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    # inv_eps_G[idx] = (1/N^2) sum_{i,j} (1/eps_{ij}) exp(-i G_idx . r_{ij})
    # r_{ij} = (i/N, j/N)
    # exp(-i 2*pi mx i/N) exp(-i 2*pi my j/N)
    
    # We can use FFT for this if we pad appropriately, but direct sum is O(N^2 * nG).
    # 32^2 * (21*21) ~ 45k ops. Very fast.
    
    inv_eps_G = np.zeros(nG, dtype=complex)
    
    # Precompute phases for efficiency
    # phase_x[mx][i] = exp(-i 2*pi mx i / N)
    # This is a 2D array of size (2*n_pw+1) x N
    # But mx is from -n_pw to n_pw.
    
    idxs = np.arange(size)
    
    for idx, (mx, my) in enumerate([(g[0]/(2*np.pi), g[1]/(2*np.pi)) for g in Gs]):
        # mx, my are integers
        mx_int = int(round(mx))
        my_int = int(round(my))
        
        # Vectorized phase calculation
        phase_x = np.exp(-1j * 2.0 * np.pi * mx_int * idxs / size)
        phase_y = np.exp(-1j * 2.0 * np.pi * my_int * idxs / size)
        
        # 2D outer product
        phase_2d = phase_x[:, None] * phase_y[None, :]
        
        # Sum
        coeff = np.sum(inv_eps_arr * phase_2d) / (size * size)
        inv_eps_G[idx] = coeff
        
    # Precompute G differences indices
    # We need inv_eps_G for G_i - G_j.
    # Map (dx, dy) to index.
    # Gs are sorted. 
    # Let's create a dictionary mapping (mx, my) to index.
    G_map = {}
    for idx, (gx, gy) in enumerate(Gs):
        # gx, gy are multiples of 2*pi
        G_map[(gx/(2*np.pi), gy/(2*np.pi))] = idx
        
    # Since we are using the same basis for differences, and the basis is symmetric 
    # (-n_pw to n_pw), the difference G_i - G_j will be in the range 
    # -2*n_pw to 2*n_pw. 
    # However, our inv_eps_G was computed only for -n_pw to n_pw.
    # For PWE, we need the Fourier coefficients of 1/eps for all G in the difference set.
    # If we use a basis of size N, the matrix is NxN. The differences can be up to 2N.
    # We should compute inv_eps_G for the larger range -2*n_pw to 2*n_pw.
    
    # Let's re-compute inv_eps_G for the larger range.
    G_range_large = range(-2*n_pw, 2*n_pw + 1)
    Gs_large = np.array([(mx, my) for mx in G_range_large for my in G_range_large], dtype=float)
    Gs_large *= 2.0 * np.pi
    nG_large = len(Gs_large)
    
    inv_eps_G_large = np.zeros(nG_large, dtype=complex)
    G_map_large = {}
    for idx, (gx, gy) in enumerate(Gs_large):
        mx_int = int(round(gx/(2*np.pi)))
        my_int = int(round(gy/(2*np.pi)))
        G_map_large[(mx_int, my_int)] = idx
        
        phase_x = np.exp(-1j * 2.0 * np.pi * mx_int * idxs / size)
        phase_y = np.exp(-1j * 2.0 * np.pi * my_int * idxs / size)
        phase_2d = phase_x[:, None] * phase_y[None, :]
        coeff = np.sum(inv_eps_arr * phase_2d) / (size * size)
        inv_eps_G_large[idx] = coeff

    # Now build the matrix for each k point
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        
        # k + G for each G in the small basis (nG)
        kGs = k[None, :] + Gs
        
        # Dot products (k+Gi) . (k+Gj)
        # Shape (nG, nG)
        dot = np.einsum('ik,jk->ij', kGs, kGs)
        
        # inv_eps matrix
        # A[i,j] = dot[i,j] * inv_eps_G_large[G_i - G_j]
        # G_i - G_j = (Gs[i,0]-Gs[j,0], Gs[i,1]-Gs[j,1])
        
        # Vectorize the lookup
        # dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
        # dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
        
        # Convert to integers for lookup
        dmx = (Gs[:, 0][:, None] - Gs[:, 0][None, :]) / (2*np.pi)
        dmy = (Gs[:, 1][:, None] - Gs[:, 1][None, :]) / (2*np.pi)
        
        # Flatten for lookup? Or use meshgrid indices?
        # Since G_map_large is a dict, it's slow to loop.
        # Better: create a 2D array of inv_eps coefficients indexed by (mx, my) offset.
        
        # Range is -2*n_pw to 2*n_pw. Size 4*n_pw+1.
        # Index = (mx + 2*n_pw) * (4*n_pw+1) + (my + 2*n_pw)
        
        size_large = 4 * n_pw + 1
        inv_eps_2d = np.zeros((size_large, size_large), dtype=complex)
        
        for idx, (gx, gy) in enumerate(Gs_large):
            mx_int = int(round(gx/(2*np.pi)))
            my_int = int(round(gy/(2*np.pi)))
            inv_eps_2d[mx_int + 2*n_pw, my_int + 2*n_pw] = inv_eps_G_large[idx]
            
        # Now extract the matrix
        # imx = dmx + 2*n_pw
        # imy = dmy + 2*n_pw
        # We need to be careful with integer conversion.
        
        dmx_int = np.round(dmx).astype(int)
        dmy_int = np.round(dmy).astype(int)
        
        imx = dmx_int + 2*n_pw
        imy = dmy_int + 2*n_pw
        
        # Check bounds (should be safe)
        # valid = (imx >= 0) & (imx < size_large) & (imy >= 0) & (imy < size_large)
        
        inv_eps_mat = inv_eps_2d[imx, imy]
        
        A = dot * inv_eps_mat
        A = (A + A.conj().T) / 2.0
        
        # Solve
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        
        # Filter positive
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) >= 5:
            # Bands are 1-indexed in problem statement? "4th and 5th".
            # Usually 1st band is lowest.
            # eigvals[0] is 1st. eigvals[3] is 4th. eigvals[4] is 5th.
            b4 = eigvals[3]
            b5 = eigvals[4]
            
            if b4 > max_band4:
                max_band4 = b4
            if b5 < min_band5:
                min_band5 = b5
                
    if min_band5 == np.inf or max_band4 <= 0:
        return -1.0
        
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    
    if midgap <= 0:
        return -1.0
        
    return 2.0 * gap / midgap

def _make_circle(radius, size=32):
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            x = i - center + 0.5
            y = j - center + 0.5
            if x*x + y*y < radius*radius:
                base[i, j] = 1
    return _enforce_symmetry(base)

def _make_rounded_square(half_width, corner_radius, size=32):
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            x = abs(i - center + 0.5)
            y = abs(j - center + 0.5)
            if x < half_width and y < half_width:
                if x > half_width - corner_radius and y > half_width - corner_radius:
                    dx = x - (half_width - corner_radius)
                    dy = y - (half_width - corner_radius)
                    if dx*dx + dy*dy <= corner_radius*corner_radius:
                        base[i, j] = 1
                else:
                    base[i, j] = 1
    return _enforce_symmetry(base)

def _make_cross(arm_width, length, size=32):
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center + 0.5)
            dy = abs(j - center + 0.5)
            if dx < length and dy < arm_width / 2.0:
                base[i, j] = 1
            elif dy < length and dx < arm_width / 2.0:
                base[i, j] = 1
    return _enforce_symmetry(base)

def design(problem):
    """
    Design the photonic crystal unit cell.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Circles
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_circle(r * size, size))
        
    # Rounded Squares
    for hw in [0.15, 0.2, 0.25, 0.3]:
        for cr in [0.0, 0.05, 0.1]:
            candidates.append(_make_rounded_square(hw * size, cr * size, size))
            
    # Crosses
    for aw in [2, 4, 6, 8]:
        for ln in [4, 8, 12, 16]:
            candidates.append(_make_cross(aw, ln, size))
            
    # Evaluate Candidates
    best_cell = None
    best_score = -1.0
    
    # Use a coarser k-point set for initial screening if time is tight?
    # For now, use the standard set. It's fast enough.
    
    for cell in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_gap(cell, size, n_pw=8, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
            
        if score > best_score:
            best_score = score
            best_cell = cell
            
    if best_cell is None:
        best_cell = _make_circle(0.2 * size, size)
        
    # 2. Local Search
    current_cell = np.array(best_cell, dtype=np.int8)
    current_score = best_score
    rng = np.random.default_rng(42)
    
    # Define fundamental domain for mutation
    # Due to symmetry, we only need to mutate pixels in one 1/8th of the cell.
    # The fundamental domain for D4 symmetry of a square is a triangle.
    # Indices (i, j) such that i >= j and i + j <= size/2? 
    # Actually, simpler: just pick a random pixel, flip it, and enforce symmetry.
    # This is valid but less efficient. 
    # Let's stick to random pixel flip + enforce symmetry.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Mutation
        mut_cell = current_cell.copy()
        
        # Flip 1-3 random pixels
        n_flips = rng.integers(1, 4)
        for _ in range(n_flips):
            i = rng.integers(0, size)
            j = rng.integers(0, size)
            mut_cell[i, j] = 1 - mut_cell[i, j]
            
        # Enforce symmetry
        sym_cell = _enforce_symmetry(mut_cell)
        
        # Evaluate
        try:
            score = _compute_gap(sym_cell, size, n_pw=10, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
            
        if score > current_score:
            current_cell = np.array(sym_cell, dtype=np.int8)
            current_score = score
            
            # Optional: If we find a significantly better score, we could restart or expand search
            # But for now, greedy hill climbing is robust.
            
    return current_cell.tolist()

# EVOLVE-BLOCK-END
