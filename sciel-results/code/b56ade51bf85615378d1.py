"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_pwe_gap(cell, eps_low=1.0, eps_high=12.0, n_pw=13):
    """
    Fast PWE surrogate using FFT to get Fourier coefficients.
    Returns the gap-midgap ratio for bands 4 and 5.
    Uses fewer plane waves for speed.
    """
    N = len(cell)
    eps = np.array(cell, dtype=float)
    eps = eps * (eps_high - eps_low) + eps_low
    
    inv_eps = 1.0 / eps
    F_inv_eps = np.fft.fft2(inv_eps) / (N * N)
    
    m_max = (n_pw - 1) // 2
    G_vectors = []
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            G_vectors.append((m, n))
    G_vectors = np.array(G_vectors)
    n_g = len(G_vectors)
    
    # Key k-points in IBZ (0 <= ky <= kx <= pi)
    k_points = [
        (0.0, 0.0),          # Gamma
        (np.pi, 0.0),        # X
        (np.pi, np.pi),      # M
        (np.pi/2, 0.0),      # Midpoint Gamma-X
        (np.pi/2, np.pi/2),  # Center
        (3*np.pi/4, np.pi/4),# Interior
        (np.pi, np.pi/2),    # Midpoint X-M
        (np.pi*0.75, np.pi*0.25),  # Interior
        (np.pi*0.6, np.pi*0.3),    # Interior
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        KG = G_vectors + np.array([kx, ky])
        M = np.zeros((n_g, n_g), dtype=complex)
        
        for i_g in range(n_g):
            for j_g in range(n_g):
                dm = G_vectors[i_g, 0] - G_vectors[j_g, 0]
                dn = G_vectors[i_g, 1] - G_vectors[j_g, 1]
                f_diff = F_inv_eps[dm % N, dn % N]
                dot = KG[i_g, 0] * KG[j_g, 0] + KG[i_g, 1] * KG[j_g, 1]
                M[i_g, j_g] = dot * f_diff / (2 * np.pi) ** 2
                
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals.real)
        
        if len(eigvals) > 4:
            w4 = np.sqrt(max(eigvals[3], 0))
            w5 = np.sqrt(max(eigvals[4], 0))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
    if min_omega_5 == np.inf:
        return 0.0
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def _make_square_cell(size, half, w, h, cx, cy):
    """Create a symmetric cell with rectangular dielectric blocks in each quadrant."""
    quadrant = np.zeros((half, half), dtype=int)
    for i in range(half):
        for j in range(half):
            x = i + 0.5
            y = j + 0.5
            if (x - cx)**2 / (w/2)**2 + (y - cy)**2 / (h/2)**2 < 1.0:
                quadrant[i][j] = 1
    cell = np.zeros((size, size), dtype=int)
    cell[:half, :half] = quadrant
    cell[:half, half:] = quadrant[:, ::-1]
    cell[half:, :half] = quadrant[::-1, :]
    cell[half:, half:] = quadrant[::-1, ::-1]
    return cell

def design(problem):
    """
    Design a 32x32 unit cell. Tries multiple design families:
    1. Circular rods (original)
    2. Elliptical/square-ish dielectric regions
    3. Ring/annulus patterns
    """
    size = problem["pixels"]
    half = size // 2
    
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_cell = None
    
    def try_cell(cell):
        nonlocal best_score, best_cell
        score = _fast_pwe_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        return score
    
    # Strategy 1: Optimize circular rods (original approach, refined)
    for r in np.arange(3.5, 7.5, 0.5):
        if time.time() - start_time > time_limit:
            break
        for cx in np.arange(5.5, 9.5, 0.5):
            if time.time() - start_time > time_limit:
                break
            quadrant = np.zeros((half, half), dtype=int)
            for i in range(half):
                for j in range(half):
                    x = i + 0.5
                    y = j + 0.5
                    if (x - cx)**2 + (y - cx)**2 < r**2:
                        quadrant[i][j] = 1
            cell = np.zeros((size, size), dtype=int)
            cell[:half, :half] = quadrant
            cell[:half, half:] = quadrant[:, ::-1]
            cell[half:, :half] = quadrant[::-1, :]
            cell[half:, half:] = quadrant[::-1, ::-1]
            try_cell(cell)
    
    # Strategy 2: Elliptical dielectric regions (wider in one direction)
    for w in np.arange(4.0, 8.0, 0.5):
        if time.time() - start_time > time_limit:
            break
        for h in np.arange(4.0, 8.0, 0.5):
            if time.time() - start_time > time_limit:
                break
            for cx in np.arange(6.0, 9.0, 0.5):
                if time.time() - start_time > time_limit:
                    break
                cell = _make_square_cell(size, half, w, h, cx, cx)
                try_cell(cell)
    
    # Strategy 3: Ring/annulus - dielectric ring in each quadrant
    for r_inner in np.arange(2.0, 5.0, 0.5):
        if time.time() - start_time > time_limit:
            break
        for r_outer in np.arange(r_inner + 1.5, r_inner + 5.0, 0.5):
            if time.time() - start_time > time_limit:
                break
            quadrant = np.zeros((half, half), dtype=int)
            for i in range(half):
                for j in range(half):
                    x = i + 0.5
                    y = j + 0.5
                    dist = np.sqrt((x - 7.5)**2 + (y - 7.5)**2)
                    if r_inner <= dist <= r_outer:
                        quadrant[i][j] = 1
            cell = np.zeros((size, size), dtype=int)
            cell[:half, :half] = quadrant
            cell[:half, half:] = quadrant[:, ::-1]
            cell[half:, :half] = quadrant[::-1, :]
            cell[half:, half:] = quadrant[::-1, ::-1]
            try_cell(cell)
    
    # Refinement: fine-tune the best found design
    if best_cell is not None:
        # Try small perturbations
        base = best_cell.copy()
        for di in range(-1, 2):
            if time.time() - start_time > time_limit:
                break
            for dj in range(-1, 2):
                if time.time() - start_time > time_limit:
                    break
                cell = base.copy()
                # Flip a few pixels near the boundary of the dielectric region
                # This is a rough local search
                if di != 0 or dj != 0:
                    # Shift the dielectric region slightly by flipping boundary pixels
                    pass  # Skip complex perturbation for now
    
    # Convert best_cell to nested list
    if best_cell is None:
        # Fallback: simple circle
        cell = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(half):
            for j in range(half):
                x = i + 0.5
                y = j + 0.5
                if (x - 7.5)**2 + (y - 7.5)**2 < 5.0**2:
                    cell[i][j] = 1
                    cell[i][size - 1 - j] = 1
                    cell[size - 1 - i][j] = 1
                    cell[size - 1 - i][size - 1 - j] = 1
        return cell
    
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
