"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _build_eps_fourier(cell, size, Npw):
    """Build the epsilon^{-1} Fourier matrix for a given PW basis size."""
    half = (Npw - 1) // 2
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / 12.0
    eps_inv[eps <= 0.5] = 1.0
    
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Compute epsilon^{-1} Fourier coefficients using FFT for speed
    eps_F = np.zeros((N, N), dtype=complex)
    # Precompute FFT of eps_inv
    eps_inv_2d = eps_inv
    fft_eps = np.fft.fft2(eps_inv_2d)
    
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            # Wrap into [-size/2, size/2-1]
            dgx_mod = dgx % size
            dgy_mod = dgy % size
            eps_F[idx_g][idx_h] = fft_eps[dgy_mod, dgx_mod] / (size * size)
    
    return eps_F, G_vectors, N

def _quick_score(cell, size, eps_high=12.0, Npw=9, n_kpoints=6):
    """Fast approximate score using reduced plane-wave basis for local optimization."""
    eps_F, G_vectors, N = _build_eps_fourier(cell, size, Npw)
    
    # Define wave vectors in irreducible BZ
    k_points = []
    # Gamma point
    k_points.append((0.0, 0.0))
    # X point
    k_points.append((0.5, 0.0))
    # M point
    k_points.append((0.5, 0.5))
    # Midpoint Gamma-X
    k_points.append((0.25, 0.0))
    # Midpoint Gamma-M
    k_points.append((0.25, 0.25))
    # Midpoint X-M
    k_points.append((0.5, 0.25))
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        Kx = 2 * np.pi * kx
        Ky = 2 * np.pi * ky
        
        A = np.zeros((N, N), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h in range(N):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(max(eigvals[3], 0))
                omega5 = np.sqrt(max(eigvals[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    return -1.0

def _four_rod_cell(size, r1, r2, r3, r4, cx=0.25, cy=0.25):
    """Create a cell with 4 circular rods at quarter-points with C4v symmetry."""
    cell = np.zeros((size, size), dtype=int)
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Four rods at (±cx, ±cy)
    radii = [r1, r2, r3, r4]
    centers = [(cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy)]
    
    for (r, (px, py)) in zip(radii, centers):
        dist = np.sqrt((X - px)**2 + (Y - py)**2)
        cell[dist < r] = 1
    
    # Enforce exact symmetry
    cell = _enforce_symmetry(cell, size)
    return cell

def _gaussian_field_to_cell(params, size, n_gaussians=3, threshold=0.5):
    """Convert Gaussian mixture parameters to a binary cell with C4v symmetry."""
    field = np.zeros((size, size), dtype=float)
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for g in range(n_gaussians):
        amp = params[g * 4]
        width = max(abs(params[g * 4 + 1]), 0.01)
        cx = params[g * 4 + 2]
        cy = params[g * 4 + 3]
        
        # 8 symmetric copies for C4v
        for dx, dy in [(cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy),
                       (cy, cx), (-cy, cx), (cy, -cx), (-cy, -cx)]:
            field += amp * np.exp(-((X - dx)**2 + (Y - dy)**2) / (2 * width**2))
    
    cell = (field > threshold).astype(int)
    cell = _enforce_symmetry(cell, size)
    return cell

def design(problem):
    """Optimize photonic crystal using four-rod superlattice with local search."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 15s margin before 120s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Get unique symmetric positions
    unique_positions = []
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            unique_positions.append((i, j))
    
    # Start with four circular rods at quarter-points
    # Radii: r1=(0.25,0.25), r2=(-0.25,0.25), r3=(0.25,-0.25), r4=(-0.25,-0.25)
    # Due to symmetry, all four should have the same radius for C4v
    # But we can vary them slightly and let symmetry enforce
    r_init = 0.12
    cell = _four_rod_cell(size, r_init, r_init, r_init, r_init)
    
    best_cell = cell.copy()
    best_score = _quick_score(best_cell, size, 12)
    
    # Phase 1: Coarse local search on unique positions
    pos_idx = list(range(len(unique_positions)))
    iterations = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10.0:
        iterations += 1
        idx = random.choice(pos_idx)
        i, j = unique_positions[idx]
        
        old_val = cell[i][j]
        cell[i][j] = 1 - old_val
        cell = _enforce_symmetry(cell, size)
        
        score = _quick_score(cell, size, 12)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            no_improve_count = 0
        else:
            cell[i][j] = old_val
            cell = _enforce_symmetry(cell, size)
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            # Try a perturbation: flip a random cluster
            for _ in range(random.randint(3, 8)):
                idx2 = random.choice(pos_idx)
                i2, j2 = unique_positions[idx2]
                cell[i2][j2] = 1 - cell[i2][j2]
            cell = _enforce_symmetry(cell, size)
            no_improve_count = 0
        
        if time.time() > deadline - 10.0:
            break
    
    # Phase 2: Try Gaussian mixture refinement
    if time.time() < deadline - 5.0:
        n_gaussians = 2
        n_params = n_gaussians * 4
        
        # Initialize from current best cell: try to fit Gaussians
        # Start with parameters near the four-rod configuration
        params = np.array([
            1.0, 0.08, 0.25, 0.25,   # amp, width, cx, cy for first gaussian
            1.0, 0.08, 0.25, 0.0     # second gaussian (will be symmetrized)
        ])
        
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb parameters
            perturbed = params + np.random.normal(0, 0.05, n_params)
            # Clamp widths
            for g in range(n_gaussians):
                perturbed[g*4+1] = max(abs(perturbed[g*4+1]), 0.02)
                perturbed[g*4] = max(perturbed[g*4], 0.1)
                perturbed[g*4+2] = np.clip(perturbed[g*4+2], -0.4, 0.4)
                perturbed[g*4+3] = np.clip(perturbed[g*4+3], -0.4, 0.4)
            
            gauss_cell = _gaussian_field_to_cell(perturbed, size, n_gaussians)
            gauss_score = _quick_score(gauss_cell, size, 12)
            
            if gauss_score > best_score:
                best_score = gauss_score
                best_cell = gauss_cell.copy()
                params = perturbed.copy()
            else:
                # Keep params close to best
                params = params + 0.1 * (perturbed - params)
    
    # Convert back to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
