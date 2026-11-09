import numpy as np
import time

def _fourier_coefficients_exact(cell, n_pw):
    """Compute exact Fourier coefficients of the pixelated dielectric map."""
    size = cell.shape[0]
    eps = np.where(cell == 1, 12.0, 1.0)
    n = n_pw
    half = n // 2
    G_indices = np.arange(-half, half + 1)
    
    coeff = np.zeros((n, n), dtype=np.complex128)
    
    for ia, nx in enumerate(G_indices):
        for ib, ny in enumerate(G_indices):
            # Compute the exact integral for this (nx, ny)
            # sum over pixels of eps_ij * integral over pixel of exp(-i 2pi (nx x + ny y))
            integral_x = 0.0
            integral_y = 0.0
            if nx == 0:
                integral_x = 1.0 / size
            else:
                # int_0^1 exp(-i 2pi nx (x + i/size)/size) dx for pixel i
                # = exp(-i 2pi nx i/size) * int_0^{1/size} exp(-i 2pi nx u) du
                # = exp(-i 2pi nx i/size) * [exp(-i 2pi nx u)/(-i 2pi nx)]_0^{1/size}
                # = exp(-i 2pi nx i/size) * (exp(-i 2pi nx /size) - 1) / (-i 2pi nx)
                phase = np.exp(-1j * 2 * np.pi * nx / size)
                integral_x = (phase - 1) / (-1j * 2 * np.pi * nx)
            
            if ny == 0:
                integral_y = 1.0 / size
            else:
                phase_y = np.exp(-1j * 2 * np.pi * ny / size)
                integral_y = (phase_y - 1) / (-1j * 2 * np.pi * ny)
            
            # Sum over all pixels
            total = 0.0
            for i in range(size):
                phase_x_i = np.exp(-1j * 2 * np.pi * nx * i / size)
                for j in range(size):
                    phase_y_j = np.exp(-1j * 2 * np.pi * ny * j / size)
                    total += eps[i, j] * phase_x_i * phase_y_j
            
            coeff[ia, ib] = total * integral_x * integral_y
    
    return coeff, G_indices

def _solve_tm_bands(cell, n_pw=17):
    """Solve TM band structure. Returns gap-midgap ratio for bands 4 and 5."""
    size = cell.shape[0]
    coeff, G_indices = _fourier_coefficients_exact(cell, n_pw)
    n = n_pw
    half = n // 2
    
    # 30 k-points in irreducible BZ
    k_points = []
    # Along Gamma-X: (k, 0)
    for k in np.linspace(0, np.pi, 7):
        k_points.append((k, 0))
    # Along X-M: (pi, k)  
    for k in np.linspace(0, np.pi, 7):
        k_points.append((np.pi, k))
    # Interior points
    for kx in np.linspace(0.2, 0.8, 3):
        for ky in np.linspace(kx, 1.0, 3):
            k_points.append((kx * np.pi, ky * np.pi))
    
    # Deduplicate and limit to 30
    k_points = list(set([(round(kx, 8), round(ky, 8)) for kx, ky in k_points]))
    k_points = np.array(k_points[:30])
    
    Gx = G_indices * 2 * np.pi
    Gy = G_indices * 2 * np.pi
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for kx, ky in k_points:
        # Build matrix
        mat = np.zeros((n, n), dtype=np.float64)
        for m in range(n):
            Gmx = Gx[m] + kx
            Gmy = Gy[m] + ky
            for nn in range(n):
                Gnx = Gx[nn] + kx
                Gny = Gy[nn] + ky
                # eps_{G_m - G_n}
                dm = m - half
                dn = nn - half
                eps_val = coeff[m, nn].real
                if abs(eps_val) < 1e-15:
                    eps_val = 1.0
                mat[m, nn] = (Gmx * Gnx + Gmy * Gny) / eps_val
        
        eigenvalues = np.linalg.eigvalsh(mat)
        eigenvalues = np.sort(eigenvalues)
        
        if len(eigenvalues) >= 5:
            omega4 = np.sqrt(eigenvalues[3])
            omega5 = np.sqrt(eigenvalues[4])
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    
    if min_omega5 == np.inf or min_omega5 < 1e-12:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    return gap / midgap

def _apply_symmetry(cell):
    """Apply D4 symmetry: cell must satisfy cell[i][j] == cell[j][i] == cell[size-1-i][j] == cell[i][size-1-j] etc."""
    size = cell.shape[0]
    result = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            # D4 orbit of (i,j)
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, i))  # transpose
            orbit.add((size-1-i, j))  # vertical flip
            orbit.add((i, size-1-j))  # horizontal flip
            orbit.add((size-1-i, size-1-j))  # 180 rot
            orbit.add((j, size-1-i))  # 90 rot
            orbit.add((size-1-j, i))  # 270 rot
            orbit.add((size-1-j, size-1-i))  # anti-diag
            vals = [cell[ii, jj] for ii, jj in orbit]
            result[i, j] = 1 if np.mean(vals) >= 0.5 else 0
    return result

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 195  # 35s margin
    
    center = (size - 1) / 2.0
    
    # Try several initial designs
    best_cell = None
    best_score = -np.inf
    
    for radius in [0.15, 0.2, 0.25, 0.3, 0.35]:
        if time.time() > deadline - 40:
            break
        test_cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                dx = (i - center) / size
                dy = (j - center) / size
                if dx * dx + dy * dy < radius ** 2:
                    test_cell[i, j] = 1
        test_cell = _apply_symmetry(test_cell)
        
        try:
            score = _solve_tm_bands(test_cell, n_pw=13)
        except Exception:
            score = -np.inf
        
        if score > best_score:
            best_score = score
            best_cell = test_cell.copy()
    
    if best_cell is None:
        # Fallback: default rod
        test_cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                dx = (i - center) / size
                dy = (j - center) / size
                if dx * dx + dy * dy < 0.2 ** 2:
                    test_cell[i, j] = 1
        best_cell = _apply_symmetry(test_cell)
    
    # Local optimization with coordinate descent
    current_cell = best_cell.copy()
    current_score = best_score
    n_pw_opt = 11
    
    no_improve = 0
    max_no_improve = 20
    
    while time.time() < deadline - 20 and no_improve < max_no_improve:
        improved = False
        for i in range(size // 2):
            if time.time() > deadline - 20:
                break
            for j in range(i, size):
                if time.time() > deadline - 20:
                    break
                new_cell = current_cell.copy()
                new_cell[i, j] = 1 - new_cell[i, j]
                new_cell = _apply_symmetry(new_cell)
                if np.array_equal(new_cell, current_cell):
                    continue
                try:
                    score = _solve_tm_bands(new_cell, n_pw=n_pw_opt)
                except Exception:
                    continue
                if score > current_score + 1e-7:
                    current_cell = new_cell
                    current_score = score
                    improved = True
        if not improved:
            no_improve += 1
    
    result = [[int(current_cell[i, j]) for j in range(size)] for i in range(size)]
    return result
