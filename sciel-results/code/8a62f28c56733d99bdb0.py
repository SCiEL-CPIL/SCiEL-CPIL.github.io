"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _make_cell_from_params(params, size=32):
    """Generate a symmetric 32x32 cell from circle parameters in the UR quadrant.
    params: list of (cx, cy, r) for circles in the upper-right quadrant (x,y in [0,0.5]).
    Mirrored to all 4 quadrants for 4-fold symmetry about cell center.
    """
    cell = np.zeros((size, size), dtype=np.float64)
    coords = (np.arange(size) + 0.5) / size - 0.5
    x_abs = np.abs(coords)
    X, Y = np.meshgrid(x_abs, x_abs, indexing='ij')
    
    for (cx, cy, r) in params:
        dist2 = (X - cx)**2 + (Y - cy)**2
        mask = dist2 < r**2
        cell += mask.astype(np.float64)
    
    cell = (cell > 0).astype(np.float64)
    
    # Enforce exact symmetry
    cell = (cell + cell[::-1].copy() + cell[:, ::-1].copy() + 
            cell[::-1].copy()[:, ::-1].copy() + cell.T + 
            cell[::-1].copy().T + cell[:, ::-1].copy().T + 
            cell[::-1].copy()[:, ::-1].copy().T) / 8.0
    cell = (cell > 0.5).astype(np.float64)
    
    return cell

def _pw_solver(cell, n_pw=17):
    """Fast plane-wave solver for TM bands 4 and 5.
    Returns (max_omega4, min_omega5) over key k-points in irreducible BZ.
    """
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    
    inv_eps = np.where(cell > 0.5, 1.0/eps_high, 1.0/eps_low)
    
    x_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    
    n = n_pw
    indices = list(range(-n//2, n//2 + 1))
    
    # Precompute Fourier coefficients of 1/eps
    fourier_coeffs = {}
    for m in indices:
        for n_idx in indices:
            Gx = 2 * np.pi * m
            Gy = 2 * np.pi * n_idx
            phase = np.exp(-1j * (Gx * X + Gy * Y))
            fourier_coeffs[(m, n_idx)] = np.sum(inv_eps * phase) / (size * size)
    
    # K-points in irreducible BZ of square lattice
    # Key high-symmetry points and paths
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # X (pi/2, pi/2) 
    k_points.append((np.pi/2, np.pi/2))
    # M (pi, 0)
    k_points.append((np.pi, 0.0))
    # M' (pi, pi) -> equivalent to Gamma by reciprocal lattice
    # Points along Gamma-X
    for t in np.linspace(0, 1, 6):
        k_points.append((t * np.pi/2, t * np.pi/2))
    # Points along X-M
    for t in np.linspace(0, 1, 6):
        k_points.append((np.pi/2 + t*np.pi/2, np.pi/2 - t*np.pi/2))
    # Points along M-Gamma
    for t in np.linspace(0, 1, 6):
        k_points.append((np.pi - t*np.pi, 0.0))
    # Some interior points
    k_points.append((np.pi/4, np.pi/4))
    k_points.append((3*np.pi/4, np.pi/4))
    k_points.append((np.pi/3, np.pi/6))
    
    # Deduplicate
    seen = set()
    unique_kpts = []
    for kx, ky in k_points:
        key = (round(kx, 8), round(ky, 8))
        if key not in seen:
            seen.add(key)
            unique_kpts.append((kx, ky))
    k_points = unique_kpts
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    G_list = [(m, n_idx) for m in indices for n_idx in indices]
    n_pw_total = len(G_list)
    
    for kx, ky in k_points:
        # Build matrices for TM mode
        # A * c = omega^2 * B * c
        # A[i,i] = |k - G_i|^2
        # B[i,j] = inv_eps(G_i - G_j)
        
        A_diag = np.zeros(n_pw_total)
        for i, (mi, ni) in enumerate(G_list):
            kx_G = kx - 2 * np.pi * mi
            ky_G = ky - 2 * np.pi * ni
            A_diag[i] = kx_G**2 + ky_G**2
        
        # Build B matrix
        B = np.zeros((n_pw_total, n_pw_total), dtype=np.complex128)
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                dm = mi - mj
                dn = ni - nj
                if (dm, dn) in fourier_coeffs:
                    B[i, j] = fourier_coeffs[(dm, dn)]
        
        # Solve generalized eigenvalue: B^{-1} A c = omega^2 c
        # Use: B^H A c = omega^2 B^H B c ... no, just use scipy eig
        try:
            from scipy.linalg import eig
            A_mat = np.diag(A_diag).astype(np.complex128)
            vals, _ = eig(A_mat, B)
            vals_real = np.real(vals)
            vals_pos = vals_real[vals_real > 1e-10]
            vals_pos.sort()
            
            if len(vals_pos) >= 5:
                omega4 = np.sqrt(vals_pos[3])
                omega5 = np.sqrt(vals_pos[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except Exception:
            continue
    
    if min_omega5 == np.inf:
        return 0.0, np.inf
    return max_omega4, min_omega5

def _score_cell(cell):
    """Compute gap-midgap ratio for TM bands 4-5."""
    try:
        max_omega4, min_omega5 = _pw_solver(cell, n_pw=15)
        if min_omega5 > max_omega4 and max_omega4 > 0:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            return gap / midgap
        elif min_omega5 <= max_omega4:
            return -1.0
        else:
            return 0.0
    except Exception:
        return -1.0

def _initial_params(size=32):
    """Generate initial circle parameters - a rod-like design."""
    # Centered rod: circle at (0.25, 0.25) with radius ~0.2
    return [(0.25, 0.25, 0.2)]

def _perturb_params(params, scale=0.03):
    """Perturb parameters for local search."""
    new_params = []
    for (cx, cy, r) in params:
        new_cx = cx + random.gauss(0, scale)
        new_cy = cy + random.gauss(0, scale)
        new_r = r + random.gauss(0, scale)
        # Keep in valid range
        new_cx = max(0.0, min(0.5, new_cx))
        new_cy = max(0.0, min(0.5, new_cy))
        new_r = max(0.01, min(0.4, new_r))
        new_params.append((new_cx, new_cy, new_r))
    return new_params

def _mutate_params(params, n_circles_max=4):
    """Mutation: possibly add or remove a circle."""
    new_params = list(params)
    action = random.random()
    
    if action < 0.3 and len(new_params) < n_circles_max:
        # Add a new circle
        cx = random.uniform(0.0, 0.5)
        cy = random.uniform(0.0, 0.5)
        r = random.uniform(0.05, 0.2)
        new_params.append((cx, cy, r))
    elif action < 0.5 and len(new_params) > 1:
        # Remove a circle
        idx = random.randint(0, len(new_params)-1)
        new_params.pop(idx)
    
    # Perturb all
    new_params = _perturb_params(new_params, scale=0.04)
    return new_params

def design(problem):
    """Search for optimal photonic crystal unit cell using continuous parameter optimization."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 230s limit
    
    # Start with the simple rod design
    best_params = _initial_params(size)
    best_cell = _make_cell_from_params(best_params, size)
    best_score = _score_cell(best_cell)
    
    # Local search with occasional global moves
    n_local_steps = 500
    step = 0
    
    while step < n_local_steps:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Try a perturbation
        candidate_params = _perturb_params(best_params, scale=0.02)
        candidate_cell = _make_cell_from_params(candidate_params, size)
        candidate_score = _score_cell(candidate_cell)
        
        if candidate_score > best_score:
            best_params = candidate_params
            best_cell = candidate_cell
            best_score = candidate_score
        else:
            # Occasionally try a bigger mutation
            if random.random() < 0.1:
                mutant_params = _mutate_params(best_params)
                mutant_cell = _make_cell_from_params(mutant_params, size)
                mutant_score = _score_cell(mutant_cell)
                if mutant_score > best_score:
                    best_params = mutant_params
                    best_cell = mutant_cell
                    best_score = mutant_score
        
        step += 1
    
    # Final check on time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass  # We already have best_cell
    
    # Convert to nested list of 0/1
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return result
# EVOLVE-BLOCK-END
