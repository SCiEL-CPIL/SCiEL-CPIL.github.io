"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _enforce_d4_symmetry(cell):
    """Enforce D4 symmetry (rotations + mirrors) on a 32x32 cell."""
    n = len(cell)
    cell = np.array(cell, dtype=np.float64)
    # Apply all 8 D4 operations and average
    transforms = [
        cell,
        cell.T,
        cell[::-1, :],
        cell[:, ::-1],
        cell[::-1, ::-1],
        cell.T[::-1, :],
        cell.T[:, ::-1],
        cell.T[::-1, ::-1],
    ]
    avg = np.mean(transforms, axis=0)
    return (avg >= 0.5).astype(int)

def _compute_tm_gap_fast(cell, n_pw=17, n_k=12):
    """Fast TM band gap calculation using plane-wave method.
    
    Returns (gap_ratio, max_omega_4, min_omega_5) where gap_ratio = 2*(min5-max4)/(min5+max4)
    """
    n = len(cell)
    eps = np.array(cell, dtype=np.float64)
    eps = 1.0 + (eps - 1.0) * 11.0  # Convert 0/1 to 1/12
    
    # Get Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    # Compute FFT for Fourier coefficients
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_fft = np.fft.fftshift(inv_eps_fft)
    
    # Plane wave vectors: use a square grid of size n_pw x n_pw
    # G vectors in units of 2*pi/a
    g_range = np.arange(-n_pw // 2, n_pw // 2 + 1)
    g_x, g_y = np.meshgrid(g_range, g_range, indexing='ij')
    g_vecs = np.stack([g_x.ravel(), g_y.ravel()], axis=1).astype(np.float64)
    n_g = len(g_vecs)
    
    # Fourier coefficients of 1/eps
    # For a periodic function, the Fourier coefficient at G is (1/N^2) * sum_{r} f(r) exp(-i G.r)
    # In pixel space, this is the FFT
    # We need to map from pixel FFT indices to G vectors
    # The FFT gives frequencies in cycles/pixel, we need in units of 2*pi/a
    # For a unit cell of size 1, pixel p corresponds to x = p/n
    # FFT frequency k corresponds to G = k * 2*pi / (n * pixel_size) = k * 2*pi / n
    # But we want G in units of 2*pi/a where a=1, so G = k
    # So FFT index k (after fftshift) corresponds to G = k
    
    # Build a lookup: for each G vector (gx, gy), get the FFT coefficient
    # FFT index for G=(gx,gy) is (gx + n//2, gy + n//2) mod n
    coeff = np.zeros(n_g, dtype=np.complex128)
    for idx, (gx, gy) in enumerate(g_vecs):
        # Map G to FFT index
        fft_ix = int(gx) % n
        fft_iy = int(gy) % n
        coeff[idx] = inv_eps_fft[fft_ix, fft_iy] / (n * n)
    
    # K points in irreducible BZ: X-M-G-X
    # BZ: -pi <= kx, ky <= pi (in units of 2*pi/a)
    # Irreducible: 0 <= ky <= kx, kx + ky <= pi
    # Key points: G=(0,0), X=(pi,0), M=(pi,pi)
    k_points = []
    for i in range(n_k):
        t = i / (n_k - 1) if n_k > 1 else 0.0
        # Path: G -> X -> M -> G (but irreducible is G-X-M-G triangle)
        # Let's use: G(0,0) -> X(pi,0) -> M(pi,pi) -> X(pi,0) -> G(0,0)
        # Actually for square lattice, irreducible BZ is the triangle with vertices (0,0), (pi,0), (pi,pi)
        # Let's sample along the edges
        if t < 0.5:
            # G to X: (0,0) to (pi,0)
            s = t / 0.5
            k_points.append((s * np.pi, 0.0))
        else:
            # X to M: (pi,0) to (pi,pi)
            s = (t - 0.5) / 0.5
            k_points.append((np.pi, s * np.pi))
    
    max_omega_4 = -np.inf
    min_omega_5 = np.inf
    
    for kx, ky in k_points:
        # Build the matrix for the plane-wave method
        # For TM: H = (1/eps) * (k + G)^2
        # The eigenvalue equation is: sum_G' (1/eps)_{G-G'} * |k+G|^2 * c_{G'} = omega^2 * c_G
        
        # k + G vectors
        kg = np.column_stack([g_vecs[:, 0] + kx, g_vecs[:, 1] + ky])
        kg_sq = np.sum(kg ** 2, axis=1)
        
        # Build the (1/eps) matrix: H_{G,G'} = (1/eps)_{G-G'} * |k+G|^2
        # (1/eps)_{G-G'} = coeff at (G-G')
        # We need to compute the convolution
        # For efficiency, we can use the fact that this is a Toeplitz-like matrix
        # But for small n_g, just build it directly
        
        # Actually, the standard formulation is:
        # sum_{G'} (1/eps)_{G-G'} * |k+G'|^2 * c_{G'} = omega^2 * c_G
        # Wait, let me be more careful.
        # The eigenvalue equation for TM is:
        # sum_{G'} (1/eps)_{G-G'} * |k+G'|^2 * c_{G'} = omega^2 * c_G
        # No, that's not right either.
        
        # Let me use the standard formulation:
        # For TM polarization, the equation is:
        # (k + G)^2 * c_G = omega^2 * sum_{G'} eps_{G-G'} * c_{G'}
        # Or equivalently:
        # sum_{G'} (1/eps)_{G-G'} * (k + G')^2 * c_{G'} = omega^2 * c_G
        
        # I'll use the second form.
        # Build matrix A where A[i,j] = (1/eps)_{G_i - G_j} * |k + G_j|^2
        # Then A @ c = omega^2 * c
        
        # Compute (1/eps)_{G_i - G_j} for all pairs
        # This is expensive for large n_g, but n_g is small here
        a_matrix = np.zeros((n_g, n_g), dtype=np.complex128)
        for i in range(n_g):
            for j in range(n_g):
                dg = g_vecs[i] - g_vecs[j]
                # Find the FFT coefficient for this difference
                # The difference G_i - G_j corresponds to FFT index
                fft_ix = int(dg[0]) % n
                fft_iy = int(dg[1]) % n
                a_matrix[i, j] = (inv_eps_fft[fft_ix, fft_iy] / (n * n)) * kg_sq[j]
        
        # Solve eigenvalue problem
        # Use np.linalg.eigvalsh for Hermitian matrices
        # The matrix should be Hermitian if the problem is set up correctly
        # Let's symmetrize to be safe
        a_matrix = 0.5 * (a_matrix + a_matrix.conj().T)
        
        eigvals = np.linalg.eigvalsh(a_matrix)
        eigvals = np.sort(eigvals.real)
        
        # Take the 5th and 4th eigenvalues (0-indexed: 3 and 4)
        if len(eigvals) >= 5:
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
    
    if max_omega_4 > 0 and min_omega_5 > 0:
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    else:
        gap_ratio = -1.0
    
    return gap_ratio, max_omega_4, min_omega_5

def _make_design_circle(radius):
    """Create a centered circle design with given radius."""
    size = 32
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            row.append(1 if x * x + y * y < radius ** 2 else 0)
        cell.append(row)
    return cell

def _make_design_ring(inner_r, outer_r):
    """Create a ring design."""
    size = 32
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            row.append(1 if inner_r ** 2 <= r2 <= outer_r ** 2 else 0)
        cell.append(row)
    return cell

def _make_design_holes(n_holes, hole_radius):
    """Create a design with holes in a dielectric background.
    
    For a square lattice, place holes at the corners and center of the unit cell.
    """
    size = 32
    # Start with all dielectric
    cell = [[1] * size for _ in range(size)]
    
    # Hole positions: for D4 symmetry, we can have holes at:
    # - Center (0,0)
    # - Corners (0.5, 0.5), (0.5, -0.5), etc. - but these are equivalent under symmetry
    # - Edge midpoints (0.5, 0), (0, 0.5), etc.
    
    # Let's try: one hole at center, and optionally holes at edge midpoints
    hole_positions = [(0.5, 0.5)]  # In cell coordinates (0 to 1)
    
    # Add edge midpoint holes for more complex patterns
    if n_holes >= 2:
        hole_positions.extend([(0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 0.5)])
    
    for hx, hy in hole_positions:
        for i in range(size):
            x = (i + 0.5) / size - 0.5 + 0.5  # Convert to 0-1 coordinate
            for j in range(size):
                y = (j + 0.5) / size - 0.5 + 0.5
                dx = x - hx
                dy = y - hy
                # Wrap around for periodicity
                if dx > 0.5: dx -= 1.0
                if dx < -0.5: dx += 1.0
                if dy > 0.5: dy -= 1.0
                if dy < -0.5: dy += 1.0
                if dx * dx + dy * dy < hole_radius ** 2:
                    cell[i][j] = 0
    
    return cell

def _make_design_multi_ring(rings):
    """Create a design with multiple concentric rings.
    
    rings: list of (inner_r, outer_r, value) tuples
    value: 1 for dielectric, 0 for air
    """
    size = 32
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            val = 0  # Default to air
            for inner_r, outer_r, v in rings:
                if inner_r ** 2 <= r2 <= outer_r ** 2:
                    val = v
            row.append(val)
        cell.append(row)
    return cell

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize TM band gap between bands 4 and 5."""
    start_time = time.time()
    deadline = start_time + 180.0  # Leave 50s safety margin
    
    size = problem["pixels"]
    
    # Generate candidate designs from several parametric families
    candidates = []
    
    # Family 1: Centered circles with various radii
    for r in np.linspace(0.1, 0.45, 15):
        candidates.append(_make_design_circle(r))
    
    # Family 2: Rings
    for inner in np.linspace(0.1, 0.3, 8):
        for outer in np.linspace(inner + 0.05, 0.45, 8):
            if outer <= inner:
                continue
            candidates.append(_make_design_ring(inner, outer))
    
    # Family 3: Hole patterns (dielectric background with air holes)
    for n_h in [1, 2]:
        for hr in np.linspace(0.05, 0.25, 10):
            candidates.append(_make_design_holes(n_h, hr))
    
    # Family 4: Multi-ring patterns
    multi_rings = [
        [(0.0, 0.15, 1), (0.15, 0.25, 0), (0.25, 0.35, 1)],
        [(0.0, 0.1, 1), (0.1, 0.2, 0), (0.2, 0.3, 1), (0.3, 0.4, 0)],
        [(0.0, 0.2, 0), (0.2, 0.3, 1)],
        [(0.0, 0.1, 0), (0.1, 0.2, 1), (0.2, 0.3, 0), (0.3, 0.4, 1)],
    ]
    for rings in multi_rings:
        candidates.append(_make_design_multi_ring(rings))
    
    # Evaluate all candidates
    best_cell = None
    best_gap = -np.inf
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Enforce symmetry
        cell_sym = _enforce_d4_symmetry(cell)
        
        # Compute gap
        try:
            gap, _, _ = _compute_tm_gap_fast(cell_sym, n_pw=13, n_k=8)
        except Exception:
            continue
        
        if gap > best_gap:
            best_gap = gap
            best_cell = [row[:] for row in cell_sym]
    
    # If no valid candidate found, fall back to the seed design
    if best_cell is None:
        return _make_design_circle(0.2)
    
    return best_cell

# EVOLVE-BLOCK-END
