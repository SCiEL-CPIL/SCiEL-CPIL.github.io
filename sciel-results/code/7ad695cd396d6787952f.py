"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _d4_symmetrize(cell):
    """Enforce D4 symmetry (transpose + 180-degree rotation) on a 32x32 binary cell."""
    n = cell.shape
    # Average over the 4 symmetry operations: identity, transpose, 180-rot, both
    sym = (cell + cell.T + cell[::-1, ::-1] + cell.T[::-1, ::-1]) / 4.0
    return (sym >= 0.5).astype(int)

def _fourier_coeffs_2d(cell, n_pw):
    """
    Compute Fourier coefficients of the inverse permittivity for a 2D periodic pixel map.
    cell: 32x32 array of 0/1 (0=air eps=1, 1=dielectric eps=12)
    n_pw: number of plane waves in each dimension (total n_pw^2)
    Returns: 2D array of shape (n_pw, n_pw) with coefficients G_x, G_y indexing.
    """
    n = cell.shape[0]
    eps = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    # Pixel centers in unit-cell coordinates [0,1)
    xs = (np.arange(n) + 0.5) / n
    ys = (np.arange(n) + 0.5) / n
    
    # Build G vectors: for a square lattice with lattice constant a=1,
    # G = (2*pi/m, 2*pi/n) for integers m, n
    # We use n_pw plane waves, so m, n range from -(n_pw//2) to (n_pw//2 - 1) (or similar)
    # Standard: use indices -N/2 to N/2-1 for even N, or -N//2 to N//2-1
    half = n_pw // 2
    g_vals = np.arange(-half, half)  # e.g., for n_pw=17, -8..8 (17 values)
    # Actually for 289 = 17^2, n_pw=17, half=8, g_vals = -8..8 (17 values)
    
    # Compute exp(-i * G . r) for each pixel and each G
    # G = 2*pi*(m, n)
    # r = (x, y)
    # For each G index (mi, ni), G = 2*pi*(g_vals[mi], g_vals[ni])
    
    # Vectorized: for each pixel (i,j) and each G pair (mi,ni)
    # phase = -2*pi*(g_vals[mi]*x_i + g_vals[ni]*y_j)
    
    # Build full arrays
    X, Y = np.meshgrid(xs, ys, indexing='ij')  # shape (n, n)
    
    # For each G vector, compute the sum over pixels of inv_eps * exp(-i*G.r) * pixel_area
    # pixel_area = (1/n)^2
    da = (1.0 / n) ** 2
    
    coeffs = np.zeros((n_pw, n_pw), dtype=complex)
    
    for mi, gm in enumerate(g_vals):
        for ni, gn in enumerate(g_vals):
            # G = 2*pi*(gm, gn)
            phase = -2.0 * np.pi * (gm * X + gn * Y)
            coeffs[mi, ni] = da * np.sum(inv_eps * np.exp(1j * phase))
    
    return coeffs

def _pwe_tm_bands(cell, n_pw=17, k_points=None):
    """
    Compute TM band structure using plane-wave expansion.
    For TM polarization, the eigenvalue equation is:
    sum_G' [ (k + G)^2 * delta_{G,G'} - (G.G')/eps(G-G') ] E_{G'} = (omega/c)^2 * E_G
    
    Actually for 2D TM (Ez field), the equation is:
    sum_G' (k+G)·(k+G') / eps(G-G') E_{G'} = (omega/c)^2 E_G
    
    Wait, let me be more careful. The standard PWE for 2D TM:
    The field Ez satisfies: (1/eps(r)) ∇² Ez = - (omega/c)^2 Ez
    Expanding in plane waves: Ez = sum_G E_G exp(i(k+G)·r)
    This gives: sum_G' (1/eps(G-G')) (k+G')² E_{G'} = (omega/c)^2 E_G
    
    So the matrix is: M_{G,G'} = (1/eps(G-G')) * (k+G')²
    And we solve M E = (omega/c)^2 E
    """
    if k_points is None:
        k_points = []
    
    n = cell.shape[0]
    half = n_pw // 2
    g_vals = np.arange(-half, half)
    
    # Precompute Fourier coefficients of 1/eps
    inv_eps_coeffs = _fourier_coeffs_2d(cell, n_pw)
    
    # G vectors: G_m = 2*pi*(g_vals[m], 0) for x-component, etc.
    # Actually G is a 2D vector: G = 2*pi*(m, n) where m, n are integers
    # For a square lattice, G = (2*pi*m/a, 2*pi*n/a) with a=1
    
    # Precompute (k+G)^2 for all G
    # G_x = 2*pi*g_vals, G_y = 2*pi*g_vals
    
    bands = []
    
    for kx, ky in k_points:
        # (k+G)^2 = (kx + 2*pi*gx)^2 + (ky + 2*pi*gy)^2
        # Build the matrix M where M[mi,ni, mj,nj] = inv_eps(G_mi-mj, G_ni-nj) * (k+G_mj,nj)^2
        
        # For efficiency, build a 4D index
        # inv_eps_coeffs[di, dj] where di = mi-mj, dj = ni-nj (mod n_pw)
        
        # (k+G)^2 for each G index
        kg_sq = np.zeros((n_pw, n_pw))
        for mi, gx in enumerate(g_vals):
            for ni, gy in enumerate(g_vals):
                kg_sq[mi, ni] = (kx + 2*np.pi*gx)**2 + (ky + 2*np.pi*gy)**2
        
        # Build matrix M: M[(mi,ni), (mj,nj)] = inv_eps_coeffs[(mi-mj) mod n_pw, (ni-nj) mod n_pw] * kg_sq[mj, nj]
        M = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
        
        for mi in range(n_pw):
            for ni in range(n_pw):
                for mj in range(n_pw):
                    for nj in range(n_pw):
                        di = (mi - mj) % n_pw
                        dj = (ni - nj) % n_pw
                        M[mi*n_pw + ni, mj*n_pw + nj] = inv_eps_coeffs[di, dj] * kg_sq[mj, nj]
        
        # Solve eigenvalue problem
        try:
            eigvals, eigvecs = np.linalg.eig(M)
            # Keep only real, positive eigenvalues
            eigvals_real = np.real(eigvals)
            mask = (eigvals_real > 0) & (np.abs(np.imag(eigvals)) < 1e-6 * np.abs(eigvals_real))
            eigvals_pos = eigvals_real[mask]
            eigvals_pos = np.sort(eigvals_pos)
            # omega^2/c^2 = eigenvalue, so omega/c = sqrt(eigenvalue)
            # We work in units where c=1, a=1, so omega = sqrt(eigenvalue)
            omega = np.sqrt(eigvals_pos)
            bands.append(omega)
        except Exception:
            continue
    
    return bands

def design(problem):
    """
    Design a D4-symmetric 32x32 photonic crystal unit cell with a large TM band gap
    between the 4th and 5th bands. Uses a fast internal PWE solver to optimize
    a parameterized family of D4-symmetric designs.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 220  # Leave 10s safety margin
    
    # Define a set of k-points in the irreducible Brillouin zone
    # For a square lattice, the irreducible BZ is 0 <= kx <= ky, kx+ky <= 1 (in units of 2*pi/a)
    # Actually the BZ is -pi <= kx,ky <= pi. Irreducible wedge: 0 <= kx <= ky, kx <= pi, ky <= pi, kx+ky <= pi
    # In units where G = 2*pi, the BZ boundary is at k = pi (in 2*pi units, that's 0.5)
    # Let me use k in units of 2*pi/a. Then BZ is -0.5 <= kx,ky <= 0.5.
    # Irreducible: 0 <= kx <= ky, kx + ky <= 0.5
    
    # Generate k-points
    k_points = []
    # High-symmetry points
    k_points.append((0.0, 0.0))       # Gamma
    k_points.append((0.5, 0.0))       # X
    k_points.append((0.5, 0.5))       # M
    k_points.append((0.25, 0.25))     # Center of X-M edge
    
    # Interior points
    n_k = 5
    for i in range(n_k):
        for j in range(n_k):
            kx = 0.5 * i / (n_k - 1)
            ky = 0.5 * j / (n_k - 1)
            if kx <= ky and kx + ky <= 0.5:
                k_points.append((kx, ky))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    k_points = unique_k
    
    def _score_cell(cell):
        """Compute the gap-midgap ratio for a given cell."""
        bands = _pwe_tm_bands(cell, n_pw=17, k_points=k_points)
        if len(bands) == 0:
            return -1.0
        
        # For each k-point, get band 4 and band 5 (0-indexed: 3 and 4)
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for omega in bands:
            if len(omega) >= 5:
                max_omega4 = max(max_omega4, omega[3])
                min_omega5 = min(min_omega5, omega[4])
        
        if min_omega5 == float('inf'):
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # Start with the seed design: centered dielectric rod
    def _make_rod(radius):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < radius*radius:
                    cell[i, j] = 1
        return cell
    
    # Try a few radii and pick the best
    best_cell = None
    best_score = -1.0
    
    radii = [0.15, 0.2, 0.25, 0.3, 0.35]
    for r in radii:
        if time.time() > deadline:
            break
        cell = _make_rod(r)
        score = _score_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If we have time, try some local perturbations
    if best_cell is not None and time.time() < deadline - 30:
        # Try flipping some pixels in a D4-symmetric way
        # The fundamental domain for D4 is the triangle 0 <= i <= j <= 15 (approximately)
        # Actually for a 32x32 grid, the D4 orbit of pixel (i,j) has up to 8 members
        # (i,j), (j,i), (31-i,j), (31-j,i), (i,31-j), (j,31-i), (31-i,31-j), (31-j,31-i)
        
        # Try flipping pixels in the fundamental domain
        n_trials = 50
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            # Pick a random pixel in the upper-left triangle of the fundamental domain
            # Fundamental domain: i <= j, i <= 15 (by symmetry about center)
            # Actually the D4 fundamental domain for a 32x32 grid centered at (15.5, 15.5)
            # is the set of pixels where i <= j and i + j <= 31 (or similar)
            # Let's just pick random pixels and enforce symmetry
            
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
            
            # Get the D4 orbit of (i, j)
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, i))
            orbit.add((31-i, j))
            orbit.add((31-j, i))
            orbit.add((i, 31-j))
            orbit.add((j, 31-i))
            orbit.add((31-i, 31-j))
            orbit.add((31-j, 31-i))
            
            # Flip all pixels in the orbit
            new_cell = best_cell.copy()
            for (oi, oj) in orbit:
                new_cell[oi, oj] = 1 - new_cell[oi, oj]
            
            score = _score_cell(new_cell)
            if score > best_score:
                best_score = score
                best_cell = new_cell
    
    # Convert to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result

# EVOLVE-BLOCK-END
