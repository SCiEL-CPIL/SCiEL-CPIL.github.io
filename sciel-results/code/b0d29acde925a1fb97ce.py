"""Photonic crystal design for large TM band gap (bands 4-5)."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _compute_band_structure_fast(cell, eps_low, eps_high):
    """Fast TM band structure using reduced plane-wave basis.
    
    Returns (max_omega_4, min_omega_5) for gap calculation.
    Uses a smaller basis for speed.
    """
    size = len(cell)
    
    # Compute Fourier coefficients of eps^{-1} for binary material
    inv_eps_low = 1.0 / eps_low
    inv_eps_high = 1.0 / eps_high
    
    # Use a smaller plane-wave basis for speed
    R = 6  # Reduced from 10 for speed
    
    # Precompute inv_eps Fourier coefficients
    inv_eps_coeffs = {}
    for mx in range(-R, R+1):
        for my in range(-R, R+1):
            if mx*mx + my*my > R*R:
                continue
            coeff = 0.0
            for i in range(size):
                for j in range(size):
                    val = inv_eps_high if cell[i][j] == 1 else inv_eps_low
                    phase = -2.0 * np.pi * (mx * (i + 0.5) + my * (j + 0.5)) / size
                    coeff += val * np.exp(1j * phase)
            coeff /= (size * size)
            inv_eps_coeffs[(mx, my)] = coeff
    
    # Plane wave basis
    G_vectors = []
    for mx in range(-R, R+1):
        for my in range(-R, R+1):
            if mx*mx + my*my <= R*R:
                G_vectors.append((mx * 2*np.pi, my * 2*np.pi, mx, my))
    N_pw = len(G_vectors)
    
    # Key k-points in irreducible BZ
    k_points = [
        (0, 0),           # Gamma
        (np.pi/2, 0),     # Midpoint X
        (np.pi, 0),       # X
        (np.pi, np.pi/2), # Midpoint M-X
        (np.pi, np.pi),   # M
        (np.pi/2, np.pi/2), # Midpoint Gamma-M
        (np.pi/np.sqrt(2), np.pi/np.sqrt(2)), # M
    ]
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        K = (kx, ky)
        
        # Build matrix
        A = np.zeros((N_pw, N_pw), dtype=complex)
        
        for i in range(N_pw):
            Gi = G_vectors[i]
            K_minus_Gi = (K[0] - Gi[0], K[1] - Gi[1])
            KmGi_sq = K_minus_Gi[0]**2 + K_minus_Gi[1]**2
            
            for j in range(N_pw):
                Gj = G_vectors[j]
                dmx = Gi[2] - Gj[2]
                dmy = Gi[3] - Gj[3]
                
                if (dmx, dmy) in inv_eps_coeffs:
                    A[i, j] = KmGi_sq * inv_eps_coeffs[(dmx, dmy)]
        
        # Solve
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals_real = np.real(eigvals)
            eigvals_pos = eigvals_real[eigvals_real > 1e-10]
            eigvals_pos.sort()
            
            if len(eigvals_pos) >= 5:
                omega_4 = np.sqrt(eigvals_pos[3])
                omega_5 = np.sqrt(eigvals_pos[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except:
            pass
    
    return max_omega_4, min_omega_5


def _superellipse_cell(size, exponent, radius, dielectric_bg=True):
    """Create a superellipse-shaped cell with symmetry enforced."""
    cell = [[0]*size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            if radius > 0:
                val = (abs(x/radius)**exponent + abs(y/radius)**exponent)
                inside = val < 1.0
            else:
                inside = False
            
            if dielectric_bg:
                cell[i][j] = 0 if inside else 1
            else:
                cell[i][j] = 1 if inside else 0
    
    # Enforce symmetry by averaging over D4 orbit
    result = [[0.0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            orbit = set()
            orbit.add((i, j))
            orbit.add((i, size-1-j))
            orbit.add((size-1-i, j))
            orbit.add((size-1-i, size-1-j))
            orbit.add((j, i))
            orbit.add((j, size-1-i))
            orbit.add((size-1-j, i))
            orbit.add((size-1-j, size-1-i))
            
            avg = sum(cell[oi][oj] for oi, oj in orbit) / len(orbit)
            for oi, oj in orbit:
                result[oi][oj] = avg
    
    return [[1 if result[i][j] > 0.5 else 0 for j in range(size)] for i in range(size)]


def _score_design(cell, eps_low, eps_high):
    """Compute the gap-midgap ratio."""
    max_omega_4, min_omega_5 = _compute_band_structure_fast(cell, eps_low, eps_high)
    if min_omega_5 <= 0 or max_omega_4 <= 0 or min_omega_5 < max_omega_4:
        return -1.0
    gap = min_omega_5 - max_omega_4
    midgap = min_omega_5 + max_omega_4
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap


def design(problem):
    """Design a photonic crystal unit cell for maximum TM band gap (bands 4-5)."""
    start_time = time.time()
    deadline = start_time + 215.0  # 15s safety margin
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    best_cell = None
    best_score = -1.0
    
    # Grid search over superellipse parameters
    # Dielectric background with air hole
    exponents = [2, 3, 4, 6, 8, 12]
    radii = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    
    # Evaluate candidates
    for n in exponents:
        for r in radii:
            if time.time() > deadline - 30:
                break
            
            cell = _superellipse_cell(size, n, r, dielectric_bg=True)
            
            try:
                score = _score_design(cell, eps_low, eps_high)
                if score > best_score:
                    best_score = score
                    best_cell = [row[:] for row in cell]
            except:
                pass
    
    # Local refinement
    if best_cell is not None and time.time() < deadline - 20:
        # Refine around best - try nearby parameters
        # We don't track which params gave best, so just try more values
        fine_exponents = [2, 3, 4, 5, 6, 7, 8]
        fine_radii = [0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
        
        for n in fine_exponents:
            for r in fine_radii:
                if time.time() > deadline - 10:
                    break
                cell = _superellipse_cell(size, n, r, dielectric_bg=True)
                try:
                    score = _score_design(cell, eps_low, eps_high)
                    if score > best_score:
                        best_score = score
                        best_cell = [row[:] for row in cell]
                except:
                    pass
    
    # Default if no result
    if best_cell is None:
        best_cell = _superellipse_cell(size, 4, 0.3, dielectric_bg=True)
    
    return best_cell
# EVOLVE-BLOCK-END
