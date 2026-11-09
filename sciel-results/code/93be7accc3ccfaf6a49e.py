<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A centred dielectric rod of radius 0.2 lattice constants (the textbook TM band-gap crystal)."""
    size = problem["pixels"]
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    return cell
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _fourier_coeffs(cell, Npw):
    """Compute Fourier coefficients of epsilon for the given cell."""
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    eps = np.array(cell, dtype=float) * (eps_high - eps_low) + eps_low
    
    coeffs = np.zeros((Npw, Npw), dtype=complex)
    for mx in range(-Npw//2, Npw//2 + 1):
        for my in range(-Npw//2, Npw//2 + 1):
            if mx == 0 and my == 0:
                coeffs[Npw//2, Npw//2] = np.mean(eps)
            else:
                # Compute Fourier coefficient using DFT on the pixel grid
                phase_x = 2j * np.pi * mx / size
                phase_y = 2j * np.pi * my / size
                x = np.arange(size)
                y = np.arange(size)
                phase = np.exp(1j * (phase_x * x[:, None] + phase_y * y[None, :]))
                coeffs[Npw//2 + mx, Npw//2 + my] = np.mean(eps * phase)
    return coeffs

def _build_pw_matrix(kx, ky, eps_coeffs, Npw):
    """Build the plane-wave matrix for TM polarization at wavevector (kx, ky)."""
    size = Npw
    half = Npw // 2
    
    # Plane wave indices
    g_x = np.arange(-half, half + 1)
    g_y = np.arange(-half, half + 1)
    
    # Build the matrix: M_ij = (k_i - G_j) · (k_i - G_j) * (1/eps_{i-j})
    # For TM: (k - G)^2 / eps_{G_i - G_j}
    
    kx, ky = float(kx), float(ky)
    
    M = np.zeros((size, size), dtype=complex)
    
    for i in range(size):
        gx_i = g_x[i]
        gy_i = g_y[i]
        kx_i = kx + gx_i
        ky_i = ky + gy_i
        
        for j in range(size):
            gx_j = g_x[j]
            gy_j = g_y[j]
            
            dx = gx_i - gx_j
            dy = gy_i - gy_j
            
            if dx == 0 and dy == 0:
                inv_eps = 1.0
            else:
                # Find the Fourier coefficient index
                idx_x = Npw//2 + dx
                idx_y = Npw//2 + dy
                if 0 <= idx_x < Npw and 0 <= idx_y < Npw:
                    eps_val = eps_coeffs[idx_x, idx_y]
                    inv_eps = 1.0 / eps_val if abs(eps_val) > 1e-12 else 1.0
                else:
                    inv_eps = 1.0
            
            val = (kx_i * kx_i + ky_i * ky_i) * inv_eps
            M[i, j] = val
    
    return M

def _solve_tm_bands(cell, kx, ky, Npw=17):
    """Solve for TM bands at a given wavevector."""
    eps_coeffs = _fourier_coeffs(cell, Npw)
    M = _build_pw_matrix(kx, ky, eps_coeffs, Npw)
    
    # M is Hermitian for real epsilon
    M = (M + M.conj().T) / 2
    eigenvalues = eigh(M, eigvals_only=True)
    # omega = sqrt(eigenvalue)
    omegas = np.sqrt(np.maximum(eigenvalues, 0))
    return omegas

def _evaluate_gap(cell, Npw=17):
    """Evaluate the gap-midgap ratio for the 4th-5th TM band gap."""
    # Irreducible Brillouin zone: kx from 0 to pi, ky from 0 to kx (for square lattice)
    # Sample points along the boundary and interior
    k_points = []
    n_samples = 10
    for i in range(n_samples + 1):
        t = i / n_samples
        k_points.append((np.pi * t, 0))  # X to Gamma
        k_points.append((0, np.pi * t))  # Gamma to Y
        k_points.append((np.pi * t, np.pi * t))  # M to X
        k_points.append((np.pi * (1-t), np.pi * t))  # interior
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for kx, ky in k_points:
        omegas = _solve_tm_bands(cell, kx, ky, Npw)
        if len(omegas) >= 5:
            omega4 = omegas[3]  # 4th band (0-indexed: 3)
            omega5 = omegas[4]  # 5th band (0-indexed: 4)
            min_omega5 = min(min_omega5, omega5)
            max_omega4 = max(max_omega4, omega4)
    
    if min_omega5 == float('inf'):
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap < 1e-12:
        return -1.0
    
    return 2.0 * gap / midgap

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry: cell == cell[::-1] row-wise, column-wise, and transpose."""
    size = len(cell)
    cell = np.array(cell, dtype=int)
    
    # Average over symmetry operations to enforce symmetry
    sym_cell = np.zeros_like(cell)
    count = 0
    
    # Identity
    sym_cell += cell
    count += 1
    
    # 90-degree rotation
    rot90 = np.rot90(cell, 1)
    sym_cell += rot90
    count += 1
    
    # 180-degree rotation
    rot180 = np.rot90(cell, 2)
    sym_cell += rot180
    count += 1
    
    # 270-degree rotation
    rot270 = np.rot90(cell, 3)
    sym_cell += rot270
    count += 1
    
    # Flip horizontal
    flip_h = cell[:, ::-1]
    sym_cell += flip_h
    count += 1
    
    # Flip vertical
    flip_v = cell[::-1, :]
    sym_cell += flip_v
    count += 1
    
    # Diagonal flip
    diag = cell.T
    sym_cell += diag
    count += 1
    
    # Anti-diagonal flip
    anti_diag = cell[::-1, ::-1].T
    sym_cell += anti_diag
    count += 1
    
    sym_cell = np.round(sym_cell / count).astype(int)
    return sym_cell.tolist()

def _random_symmetric_cell(size, density):
    """Generate a random symmetric cell with given density."""
    # Generate random cell in the fundamental domain (upper-right quadrant)
    half = size // 2
    cell = np.zeros((size, size), dtype=int)
    
    for i in range(half):
        for j in range(half):
            if np.random.random() < density:
                cell[i, j] = 1
    
    # Mirror to enforce symmetry
    cell[half:, :] = cell[:half, :][:, ::-1]
    cell[:, half:] = cell[:, :half][::-1, :]
    
    return cell.tolist()

def design(problem):
    """Optimize photonic crystal geometry using local search with plane-wave solver."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with the seed: a centered dielectric rod
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    
    best_cell = [row[:] for row in cell]
    best_score = _evaluate_gap(best_cell)
    
    # Local search: flip individual pixels while maintaining symmetry
    Npw = 13  # Smaller for speed during search
    
    # Precompute which pixel flips maintain symmetry
    # A pixel at (i, j) can be flipped if all its symmetry-equivalent pixels are flipped together
    # For simplicity, we'll flip orbits under the symmetry group
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random pixel to try flipping
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Compute the orbit of (i, j) under the symmetry group
        orbit = set()
        orbit.add((i, j))
        orbit.add((i, size - 1 - j))  # horizontal flip
        orbit.add((size - 1 - i, j))  # vertical flip
        orbit.add((size - 1 - i, size - 1 - j))  # 180 rotation
        orbit.add((j, i))  # transpose
        orbit.add((j, size - 1 - i))  # anti-transpose
        orbit.add((size - 1 - j, i))
        orbit.add((size - 1 - j, size - 1 - i))
        
        # Check if all pixels in the orbit have the same value
        val = cell[i][j]
        all_same = all(cell[pi][pj] == val for pi, pj in orbit)
        
        if not all_same:
            continue
        
        # Try flipping
        new_cell = [row[:] for row in cell]
        for pi, pj in orbit:
            new_cell[pi][pj] = 1 - val
        
        new_score = _evaluate_gap(new_cell, Npw)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell
        else:
            # Revert (cell was not modified)
            pass
    
    # Final evaluation with higher precision
    final_score = _evaluate_gap(best_cell, Npw=17)
    
    return best_cell
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
