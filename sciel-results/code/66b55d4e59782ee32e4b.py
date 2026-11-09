"""Design a symmetric 32x32 photonic-crystal unit cell for a large TM band gap (bands 4-5)."""
import time
import numpy as np
from numpy.fft import rfft2, irfft2

# --- Fast approximate band solver ---

def _fourier_coefficients(cell, eps_low=1.0, eps_high=12.0):
    """Compute Fourier coefficients of 1/epsilon for the pixel map."""
    N = cell.shape[0]
    eps = np.where(cell == 1, eps_high, eps_low)
    inv_eps = 1.0 / eps
    # Compute DFT of inv_eps (this gives Fourier coefficients)
    # The pixel map is periodic, so we use FFT
    F = np.fft.fft2(inv_eps)
    # Normalize: F[k] = sum_{n} inv_eps[n] exp(-2pi i k.n/N)
    # For the plane-wave expansion, we need the Fourier series coefficients
    # inv_eps(x) = sum_k c_k exp(2pi i k.x/N)
    # c_k = (1/N^2) sum_n inv_eps[n] exp(-2pi i k.n/N)
    c = F / (N * N)
    return c

def _solve_tm_bands(cell, kx, ky, num_pws=17, eps_low=1.0, eps_high=12.0):
    """
    Solve TM bands using plane-wave method.
    Returns sorted list of omega values (in units of 2pi/a).
    """
    N = cell.shape[0]
    c = _fourier_coefficients(cell, eps_low, eps_high)
    
    # Number of plane waves
    npw = num_pws
    # Generate plane wave vectors
    # For a square lattice, we use G vectors with |G| <= Gmax
    # Gmax chosen to include npw^2/4 plane waves approximately
    # Use a circular cutoff
    half = npw // 2
    gxs = np.arange(-half, half + 1)
    gys = np.arange(-half, half + 1)
    GX, GY = np.meshgrid(gxs, gys, indexing='ij')
    G2 = GX**2 + GY**2
    # Keep only those within a radius to get ~npw^2/4
    radius = half
    mask = G2 <= radius * radius
    gx = GX[mask].astype(float)
    gy = GY[mask].astype(float)
    n_pws = len(gx)
    
    # Build the TM eigenvalue matrix
    # For TM: (k - G)^2 * A[k-G, k-G'] = sum_{G'} (1/eps)_{k-G, k-G'} * ...
    # Actually for TM polarization:
    # det[(k-G)^2 delta_{G,G'} - sum_{G''} (1/eps)_{G-G''} (k-G'')(k-G'')] = 0
    # Wait, let me be more careful.
    
    # TM mode: H field. The eigenvalue equation is:
    # sum_{G'} [(k - G')^2 delta_{G,G'} - (1/eps)_{G-G'} (k-G)(k-G')] H_{G'} = omega^2 H_G
    # No wait. Let me think again.
    
    # For a 2D photonic crystal with epsilon(x,y) periodic:
    # TM polarization (H_z):
    # -nabla^2 H_z = (omega^2/c^2) epsilon(x,y) H_z
    # In plane-wave basis: H_z = sum_G H_G exp(iG.r)
    # The equation becomes:
    # sum_{G'} [ (k-G)^2 delta_{G,G'} - sum_{G''} (1/epsilon)_{G-G''} (k-G'')(k-G'') ] H_{G'} = (omega^2/c^2) H_G
    # Hmm, that's not quite right either. Let me use the standard formulation.
    
    # Standard PW method for TM:
    # The equation is: sum_{G'} [(k+G)^2 delta_{G,G'} - (1/eps)_{G-G'} (k+G)(k+G')] H_{G'} = (omega/c)^2 H_G
    # Wait no. Let me look at this more carefully.
    
    # Maxwell's equations in 2D for TM (H_z, E_x, E_y):
    # curl E = i omega mu H -> gives relations
    # The eigenvalue problem for TM is:
    # sum_{G'} [ (k-G')^2 delta_{G,G'} - (1/epsilon)_{G-G'} (k-G)(k-G') ] H_{G'} = (omega^2/c^2) H_G
    # Hmm, I think the correct form is:
    # For TM: the operator is -nabla_perp^2 / epsilon(x,y)
    # In PW: sum_{G'} [(k-G)^2 delta_{G,G'} - sum_{G''} (1/eps)_{G-G''} (k-G'')(k-G'')] ... 
    # 
    # Actually, the standard form for TM is:
    # A_{G,G'} = (k-G)^2 delta_{G,G'} - (1/eps)_{G-G'} (k-G)(k-G')
    # Wait, that doesn't look right dimensionally. Let me reconsider.
    
    # The correct TM eigenvalue equation in plane-wave expansion:
    # sum_{G'} [ (k+G)^2 delta_{G,G'} - sum_{G''} (1/epsilon)_{G-G''} (k+G)(k+G'') ] H_{G'} = (omega^2/c^2) H_G
    # No, I think it's:
    # For TM: H = sum_G H_G e^{iG.r}, and the equation is:
    # sum_{G'} [(k-G)^2 delta_{G,G'} - (1/eps)_{G-G'} (k-G)(k-G')] H_{G'} = (omega/c)^2 H_G
    # Hmm, I'm confusing myself. Let me just use the standard formulation:
    
    # For TM polarization, the eigenvalue equation is:
    # sum_{G'} A_{G,G'} H_{G'} = (omega/c)^2 H_G
    # where A_{G,G'} = (k+G)^2 delta_{G,G'} - (1/epsilon)_{G-G'} (k+G)(k+G')
    # Wait, I think the correct form involves the dot product of (k+G) and (k+G').
    
    # Let me use a simpler and correct formulation. For TM:
    # The operator is: - (1/epsilon(x)) nabla_perp^2
    # In plane-wave basis, this gives a matrix where:
    # A_{G,G'} = (k+G)^2 delta_{G,G'} - sum_{G''} (1/eps)_{G-G''} (k+G'')(k+G')
    # Hmm, that still doesn't seem right.
    
    # OK let me just use the well-known result:
    # For TM: M_{G,G'} = (k+G)^2 delta_{G,G'} - (1/eps)_{G-G'} [(k+G)_x (k+G')_x + (k+G)_y (k+G')_y]
    # No wait, that's not right either.
    
    # Let me think from scratch. The TM eigenvalue problem is:
    # -nabla^2 H_z / epsilon(r) = (omega^2/c^2) H_z
    # Write H_z = sum_G H_G exp(i G.r), 1/epsilon(r) = sum_G c_G exp(i G.r)
    # Then: -nabla^2 H_z = sum_G (k+G)^2 H_G exp(i(k+G).r)
    # And (1/epsilon) * (-nabla^2 H_z) involves convolution:
    # sum_{G,G'',G'} c_{G''} (k+G')^2 H_{G'} exp(i(G+G'+G'').r)
    # Setting G + G' + G'' = G (i.e., G'' = 0 for the output mode G):
    # sum_{G'} c_{G-G'} (k+G')^2 H_{G'}
    # So the equation is:
    # sum_{G'} c_{G-G'} (k+G')^2 H_{G'} = (omega^2/c^2) H_G
    # This is a generalized eigenvalue problem if we write it as:
    # sum_{G'} c_{G-G'} (k+G')^2 H_{G'} = (omega^2/c^2) H_G
    # But c_0 = 1 (if we normalize), so this is:
    # sum_{G'} c_{G-G'} (k+G')^2 H_{G'} = (omega^2/c^2) H_G
    # This is NOT a standard eigenvalue problem because of the (k+G')^2 factor.
    
    # Actually wait. Let me redo this. The equation is:
    # (1/epsilon(r)) * (-nabla^2 H_z) = (omega^2/c^2) H_z
    # LHS: (1/epsilon(r)) * (-nabla^2 H_z)
    # -nabla^2 H_z = sum_{G'} (k+G')^2 H_{G'} exp(i(k+G').r)
    # 1/epsilon(r) = sum_{G''} c_{G''} exp(i G''.r)
    # Product: sum_{G',G''} c_{G''} (k+G')^2 H_{G'} exp(i(k+G'+G'').r)
    # For the coefficient of exp(i(k+G).r): G' + G'' = G, so G'' = G - G'
    # LHS coefficient: sum_{G'} c_{G-G'} (k+G')^2 H_{G'}
    # RHS: (omega^2/c^2) H_G
    # So: sum_{G'} c_{G-G'} (k+G')^2 H_{G'} = (omega^2/c^2) H_G
    # This is a matrix equation: M H = (omega^2/c^2) H
    # where M_{G,G'} = c_{G-G'} (k+G')^2
    # This is a standard eigenvalue problem!
    
    # Build the matrix
    M = np.zeros((n_pws, n_pws), dtype=complex)
    for i in range(n_pws):
        for j in range(n_pws):
            # G - G' = (gx[i]-gx[j], gy[i]-gy[j])
            diff_x = int(round(gx[i] - gx[j]))
            diff_y = int(round(gy[i] - gy[j]))
            # Get c_{G-G'}
            # c is indexed by FFT indices. For a 2D FFT of size N:
            # c[kx,ky] corresponds to frequency (kx, ky) where kx,ky in [0, N)
            # But we need negative frequencies too.
            # numpy fft2 gives frequencies in order: 0, 1, ..., N/2-1, -N/2+1, ..., -1
            # So for a negative frequency -m, the index is N-m
            cidx_x = diff_x % N
            cidx_y = diff_y % N
            c_val = c[cidx_x, cidx_y]
            # (k+G')^2
            kgx = kx + gx[j]
            kgy = ky + gy[j]
            kg2 = kgx*kgx + kgy*kgy
            M[i, j] = c_val * kg2
    
    # Solve eigenvalue problem
    # M is not Hermitian in general because c_{G-G'} != c_{G'-G}^* unless the medium is real
    # But since epsilon is real, c is such that c_{-G} = c_G^*, so M should be similar to a Hermitian matrix
    # Actually, M_{G,G'} = c_{G-G'} (k+G')^2
    # M_{G',G} = c_{G'-G} (k+G)^2
    # These are not conjugates of each other in general.
    # But the eigenvalues should be real and positive.
    # Let's just use eigh if M is Hermitian, or eig otherwise.
    
    # Check if M is approximately Hermitian
    if np.allclose(M, M.conj().T, atol=1e-10):
        eigvals = np.linalg.eigvalsh(M)
    else:
        eigvals = np.linalg.eigvals(M)
        eigvals = np.real(eigvals)
    
    # Sort and return
    eigvals = np.sort(eigvals)
    # Filter out negative values (numerical artifacts)
    eigvals = eigvals[eigvals > 0]
    return eigvals

def _score_cell(cell, k_points, num_pws=17, eps_low=1.0, eps_high=12.0):
    """Score a cell by the gap-midgap ratio between bands 4 and 5."""
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        try:
            omegas = _solve_tm_bands(cell, kx, ky, num_pws, eps_low, eps_high)
            if len(omegas) >= 5:
                omega4 = omegas[3]  # 4th band (0-indexed: 3)
                omega5 = omegas[4]  # 5th band (0-indexed: 4)
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0.0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap == 0:
        return -1.0
    return 2.0 * gap / midgap

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry: cell[i][j] = cell[N-1-i][j] = cell[i][N-1-j] = cell[N-1-i][N-1-j]
    and cell[i][j] = cell[j][i]."""
    N = cell.shape[0]
    sym_cell = cell.copy()
    # Enforce all four symmetries by averaging (but since binary, we use majority or just force)
    # For binary, we'll just enforce: if any of the symmetric positions is 1, set all to 1? 
    # No, that would make everything 1. Instead, we enforce that all symmetric positions are equal.
    # We'll take the value from the first position in the orbit.
    for i in range(N):
        for j in range(N):
            # Orbit under D4: (i,j), (N-1-i,j), (i,N-1-j), (N-1-i,N-1-j), (j,i), (j,N-1-i), (N-1-j,i), (N-1-j,N-1-i)
            orbit = [(i,j), (N-1-i,j), (i,N-1-j), (N-1-i,N-1-j), 
                     (j,i), (j,N-1-i), (N-1-j,i), (N-1-j,N-1-i)]
            # Use the first one as reference
            val = cell[i][j]
            for ii, jj in orbit:
                sym_cell[ii][jj] = val
    return sym_cell

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 180  # Leave 40s margin for safety
    
    # Define k-points in the irreducible Brillouin zone
    # For a square lattice, the IBZ is: 0 <= kx <= ky, 0 <= kx, ky <= 0.5 (in units of 2pi/a)
    # Actually the first BZ is [-0.5, 0.5]^2, and the IBZ is the triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # Let's use a set of k-points
    k_points = []
    # Along Gamma-X: (0,0) to (0.5, 0)
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0))
    # Along X-M: (0.5,0) to (0.5,0.5)
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Along M-Gamma: (0.5,0.5) to (0,0)
    for t in np.linspace(0, 1, 8):
        k_points.append((0.5*(1-t), 0.5*(1-t)))
    # Some interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.3, 0.2))
    k_points.append((0.4, 0.3))
    k_points.append((0.15, 0.05))
    k_points.append((0.35, 0.15))
    
    # Remove duplicates
    k_points = list(set(k_points))
    
    # Start with the seed design
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i][j] = 1
    
    best_cell = cell.copy()
    best_score = _score_cell(cell, k_points, 17, eps_low, eps_high)
    
    # Genetic algorithm
    npop = 12
    population = [cell.copy() for _ in range(npop)]
    
    gen = 0
    while time.time() < deadline - 20:
        gen += 1
        
        # Evaluate population
        scores = []
        for ind in population:
            s = _score_cell(ind, k_points, 17, eps_low, eps_high)
            scores.append(s)
        
        # Update best
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_cell = population[best_idx].copy()
        
        # Create next generation
        new_pop = [best_cell.copy()]  # Elitism
        
        while len(new_pop) < npop:
            if time.time() > deadline - 25:
                break
            # Tournament selection
            i1 = np.random.randint(npop)
            i2 = np.random.randint(npop)
            parent = population[i1] if scores[i1] >= scores[i2] else population[i2]
            
            # Mutation: flip some random pixels
            child = parent.copy()
            n_mut = np.random.randint(1, 6)
            for _ in range(n_mut):
                mi = np.random.randint(size)
                mj = np.random.randint(size)
                child[mi, mj] = 1 - child[mi, mj]
            
            # Enforce symmetry
            child = _apply_symmetry(child)
            
            new_pop.append(child)
        
        population = new_pop
    
    # Return best
    result = best_cell.tolist()
    return result
