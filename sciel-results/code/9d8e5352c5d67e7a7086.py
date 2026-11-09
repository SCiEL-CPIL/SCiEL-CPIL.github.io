"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap between bands 4 and 5."""
import numpy as np
import time
import itertools

def _enforce_symmetry(cell):
    """Enforce 4-fold rotational and mirror symmetry about the center."""
    size = len(cell)
    cell = np.array(cell, dtype=np.int8)
    # The cell must be invariant under:
    # 1. cell == cell[::-1] row-wise (vertical mirror)
    # 2. cell == cell[::-1] column-wise (horizontal mirror)
    # 3. cell == cell.T (transpose)
    # 
    # For a 32x32 grid, the fundamental domain is the top-left 16x16 quadrant.
    # We can define the top-left 16x16 and mirror it to create the full cell.
    
    # Extract the fundamental 16x16 block (top-left quadrant)
    # For even size, the center is between pixels 15 and 16
    # Mirror: row i maps to row size-1-i, col j maps to col size-1-j
    
    # Let's define: for i in range(16), j in range(16):
    # cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
    # And cell[i][j] = cell[j][i] (transpose symmetry)
    
    # So the fundamental domain is actually the top-left 16x16, but with the constraint
    # that it must be symmetric about the diagonal within that quadrant? No.
    
    # Let me think more carefully. The constraints are:
    # A) cell[i][j] == cell[31-i][j] for all i,j (vertical flip)
    # B) cell[i][j] == cell[i][31-j] for all i,j (horizontal flip)  
    # C) cell[i][j] == cell[j][i] for all i,j (transpose)
    
    # From A and B: cell[i][j] == cell[31-i][31-j] (180 rotation)
    # From C and A: cell[i][j] == cell[j][i] == cell[31-j][i]
    # So the fundamental domain is 16x16 (top-left quadrant), and:
    # - cell[i][j] for i,j in 0..15 determines the rest
    # - But also cell[i][j] == cell[j][i] means the 16x16 block itself must be symmetric? 
    #   No, wait. C says cell[i][j] == cell[j][i] for ALL i,j, not just within the quadrant.
    #   So if i,j are both in 0..15, then cell[i][j] == cell[j][i].
    #   If i in 0..15 and j in 16..31, then cell[i][j] == cell[j][i], and cell[j][i] is in the bottom-left quadrant.
    
    # The fundamental domain for the full symmetry group D4 acting on a 32x32 grid:
    # The orbit of a pixel (i,j) under D4 has up to 8 elements.
    # For the top-left 16x16 quadrant, pixels on the diagonals have smaller orbits.
    
    # Simplest approach: define the top-left 16x16, then enforce all symmetries by averaging or taking max.
    # Actually, for binary, we just need consistency. Let's define the fundamental 16x16 and construct the rest.
    
    # For pixel (i,j) with i,j in 0..15:
    # Its orbit under D4: (i,j), (31-i,j), (i,31-j), (31-i,31-j), (j,i), (31-j,i), (j,31-i), (31-j,31-i)
    # But many of these may coincide.
    
    # Let me just construct the full cell from the 16x16 fundamental block and verify.
    fundamental = cell[:16, :16].copy()
    
    # Build full cell
    full = np.zeros((size, size), dtype=np.int8)
    for i in range(16):
        for j in range(16):
            val = fundamental[i, j]
            # All symmetric positions
            positions = set()
            positions.add((i, j))
            positions.add((31-i, j))
            positions.add((i, 31-j))
            positions.add((31-i, 31-j))
            positions.add((j, i))
            positions.add((31-j, i))
            positions.add((j, 31-i))
            positions.add((31-j, 31-i))
            for (pi, pj) in positions:
                full[pi, pj] = val
    
    return full

def _compute_fourier_coeffs(cell, n_pw):
    """Compute Fourier coefficients of the dielectric function for plane-wave method.
    
    For TM polarization, we need 1/epsilon(r).
    cell: 32x32 array of 0/1 (0=air eps=1, 1=dielectric eps=12)
    n_pw: number of plane waves in each direction
    """
    size = len(cell)
    eps = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    # Compute 2D DFT of inv_eps
    # G vectors: G = (2*pi*m/L, 2*pi*n/L) where m,n are integers
    # For a 32x32 pixel grid, we compute the discrete Fourier transform
    
    # The Fourier coefficient for G=(m,n) is:
    # eps_G = (1/N^2) * sum_{i,j} eps(i,j) * exp(-i*G.(r_i,r_j))
    # where r_i = (i+0.5)/size (pixel centers)
    
    # Using numpy FFT:
    # np.fft.fft2 gives F[m,n] = sum_{i,j} f[i,j] * exp(-2*pi*i*m*i/N) * exp(-2*pi*i*n*j/N)
    # But we need exp(-i*G.r) where G = 2*pi*(m,n)/L and r = (i+0.5)/N * L
    # So G.r = 2*pi*(m,n).((i+0.5)/N, (j+0.5)/N) = 2*pi*(m*i/N + m/(2N) + n*j/N + n/(2N))
    # = 2*pi*m*i/N + 2*pi*n*j/N + 2*pi*(m+n)/(2N)
    # = 2*pi*m*i/N + 2*pi*n*j/N + pi*(m+n)/N
    
    # So F[m,n]_correct = exp(-i*pi*(m+n)/N) * np.fft.fft2(f)[m,n] / N^2
    
    fft_inv_eps = np.fft.fft2(inv_eps)
    # Phase correction for pixel centers
    i_idx = np.arange(size)[:, None]
    j_idx = np.arange(size)[None, :]
    phase = np.exp(-1j * np.pi * (i_idx + j_idx) / size)
    # But we need this for each G index (m,n), not pixel index
    # Actually, the FFT output index (m,n) corresponds to G=(m,n) (with wraparound)
    # The phase factor is exp(-i*pi*(m+n)/N)
    
    m_idx = np.arange(size)[:, None]
    n_idx = np.arange(size)[None, :]
    phase_correction = np.exp(-1j * np.pi * (m_idx + n_idx) / size)
    
    fourier_coeffs = fft_inv_eps * phase_correction / (size * size)
    
    # Now select the n_pw x n_pw set of G vectors centered around 0
    # G indices range from -n_pw//2 to n_pw//2 - 1 (for even n_pw)
    half = n_pw // 2
    indices = list(range(-half, half))
    # Map to FFT indices
    fft_indices = [idx % size for idx in indices]
    
    selected = np.zeros((n_pw, n_pw), dtype=complex)
    for ii, mi in enumerate(fft_indices):
        for jj, ni in enumerate(fft_indices):
            selected[ii, jj] = fourier_coeffs[mi, ni]
    
    return selected

def _pw_solve_tm(kx, ky, cell, n_pw=17):
    """Solve the TM band structure at a given wavevector using plane-wave expansion.
    
    For TM: (1/eps(r)) * (-nabla x (nabla x E)) = omega^2 E
    In plane-wave form: sum_G' eps_{G-G'} * |G+k|^2 * E_{G'} = omega^2 * E_G
    
    We want the 4th and 5th eigenvalues.
    """
    size = len(cell)
    eps_G = _compute_fourier_coeffs(cell, n_pw)
    
    # G vectors
    half = n_pw // 2
    g_indices = np.array(range(-half, half))
    Gx = 2 * np.pi * g_indices / size
    Gy = 2 * np.pi * g_indices / size
    
    # Build the matrix: A[G, G'] = eps_{G-G'} * |G+k|^2
    # where eps_{G-G'} is the Fourier coefficient at G-G'
    
    # k vector
    k = np.array([kx, ky])
    
    # Precompute |G+k|^2 for all G
    Gx_grid, Gy_grid = np.meshgrid(Gx, Gy, indexing='ij')
    kx_grid = Gx_grid + kx
    ky_grid = Gy_grid + ky
    k_sq = kx_grid**2 + ky_grid**2
    
    # Build matrix
    n = n_pw
    A = np.zeros((n*n, n*n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            G_i = np.array([Gx[i], Gy[j]])
            ki_sq = (G_i[0] + kx)**2 + (G_i[1] + ky)**2
            
            for ii in range(n):
                for jj in range(n):
                    G_ii = np.array([Gx[ii], Gy[jj]])
                    dG = G_i - G_ii
                    dm = int(round(dG[0] / (2*np.pi/size)))
                    dn = int(round(dG[1] / (2*np.pi/size)))
                    
                    # Look up eps_{dm, dn}
                    # dm, dn range from -(n-1) to (n-1)
                    # Map to index in eps_G array
                    # eps_G[ii', jj'] corresponds to G index (g_indices[ii'], g_indices[jj'])
                    # We need the index where g_indices[idx] == dm
                    try:
                        idx_m = g_indices.tolist().index(dm)
                        idx_n = g_indices.tolist().index(dn)
                        eps_val = eps_G[idx_m, idx_n]
                    except ValueError:
                        eps_val = 0.0
                    
                    A[i*n+j, ii*n+jj] = eps_val * ki_sq
    
    # Solve eigenvalue problem
    # A is Hermitian (should be, for real eps), so use eigh
    # But A might not be exactly Hermitian due to numerical issues
    # Symmetrize
    A = (A + A.conj().T) / 2.0
    
    eigenvalues = np.linalg.eigvalsh(A)
    
    # eigenvalues are omega^2, sort in ascending order
    # Return the 4th and 5th eigenvalues (0-indexed: 3 and 4)
    return eigenvalues[3], eigenvalues[4]

def _score_cell(cell, k_points=None, n_pw=17):
    """Score a cell by computing the gap-midgap ratio between bands 4 and 5."""
    if k_points is None:
        # Irreducible Brillouin zone for square lattice: triangle from (0,0) to (pi,0) to (pi,pi)
        # Sample points along the boundary and interior
        k_points = []
        # Along Gamma-X: (0,0) to (pi,0)
        for t in np.linspace(0, 1, 5):
            k_points.append((np.pi * t, 0))
        # Along X-M: (pi,0) to (pi,pi)
        for t in np.linspace(0, 1, 5):
            k_points.append((np.pi, np.pi * t))
        # Along M-Gamma: (pi,pi) to (0,0)
        for t in np.linspace(0, 1, 5):
            k_points.append((np.pi * (1-t), np.pi * (1-t)))
        # Interior points
        k_points.append((np.pi/2, np.pi/2))
        k_points.append((np.pi/2, 0))
        k_points.append((0, np.pi/2))
        k_points.append((np.pi/2, np.pi/4))
        k_points.append((np.pi/4, np.pi/4))
    
    max_omega4 = 0
    min_omega5 = np.inf
    
    for (kx, ky) in k_points:
        try:
            w4_sq, w5_sq = _pw_solve_tm(kx, ky, cell, n_pw)
            w4 = np.sqrt(max(0, w4_sq))
            w5 = np.sqrt(max(0, w5_sq))
            max_omega4 = max(max_omega4, w4)
            min_omega5 = min(min_omega5, w5)
        except Exception:
            continue
    
    if max_omega4 == 0 or min_omega5 == np.inf:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap == 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _random_symmetric_cell(size=32):
    """Generate a random symmetric 32x32 cell."""
    fundamental = np.random.randint(0, 2, (16, 16))
    cell = _enforce_symmetry(fundamental)
    return cell

def _mutate_cell(cell, mutation_rate=0.1):
    """Mutate a cell by flipping random pixels while maintaining symmetry."""
    cell = cell.copy()
    size = len(cell)
    # Mutate in the fundamental domain
    for i in range(16):
        for j in range(16):
            if np.random.random() < mutation_rate:
                # Flip this pixel and all its symmetric partners
                val = 1 - cell[i, j]
                positions = [(i, j), (31-i, j), (i, 31-j), (31-i, 31-j),
                             (j, i), (31-j, i), (j, 31-i), (31-j, 31-i)]
                for (pi, pj) in set(positions):
                    cell[pi, pj] = val
    return cell

def _crossover(cell1, cell2):
    """Crossover two cells."""
    size = len(cell1)
    child = np.zeros((size, size), dtype=np.int8)
    # Random mask for crossover
    mask = np.random.randint(0, 2, (16, 16))
    for i in range(16):
        for j in range(16):
            if mask[i, j]:
                child[i, j] = cell1[i, j]
            else:
                child[i, j] = cell2[i, j]
    return _enforce_symmetry(child)

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Start with the textbook dielectric rod
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    best_cell = np.array(cell, dtype=np.int8)
    best_score = _score_cell(best_cell)
    
    # Genetic algorithm
    pop_size = 20
    population = [best_cell.copy()]
    for _ in range(pop_size - 1):
        population.append(_random_symmetric_cell(size))
    
    generation = 0
    while time.time() < deadline - 30:  # Leave 30s safety margin
        generation += 1
        
        # Score the population
        scored = []
        for cell in population:
            score = _score_cell(cell)
            scored.append((score, cell))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_cell = scored[0][1].copy()
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top 2
        for i in range(min(2, len(scored))):
            new_population.append(scored[i][1].copy())
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 40:
                break
            
            # Select parents (tournament)
            p1 = scored[np.random.randint(0, min(10, len(scored)))][1]
            p2 = scored[np.random.randint(0, min(10, len(scored)))][1]
            
            # Crossover
            child = _crossover(p1, p2)
            
            # Mutate
            mutation_rate = max(0.02, 0.15 / (1 + generation * 0.1))
            child = _mutate_cell(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # Convert to nested list
    result = best_cell.tolist()
    return result
