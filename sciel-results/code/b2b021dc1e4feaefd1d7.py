"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, quarter):
    """Build a 32x32 cell from a 16x16 top-left quarter, enforcing full square symmetry."""
    n = size // 2  # 16
    cell = []
    for i in range(size):
        ri = i if i < n else (2 * n - 1 - i)
        row = []
        for j in range(size):
            rj = j if j < n else (2 * n - 1 - j)
            row.append(quarter[ri][rj])
        cell.append(row)
    return cell


def _compute_1over_eps_fourier(cell, size, n_pw):
    """Compute Fourier coefficients of 1/epsilon for the pixel map."""
    half = n_pw // 2
    inv_eps_coeff = {}
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                for j in range(size):
                    y = (j + 0.5) / size
                    eps = 12.0 if cell[i][j] == 1 else 1.0
                    inv_eps = 1.0 / eps
                    angle = 2.0 * math.pi * (mx * x + my * y)
                    s += inv_eps * (math.cos(angle) + 1j * math.sin(angle))
            inv_eps_coeff[(mx, my)] = s / (size * size)
    return inv_eps_coeff


def _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=30):
    """Compute TM band structure using plane-wave Galerkin method."""
    half = n_pw // 2
    G_vectors = []
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            G_vectors.append((2.0 * math.pi * mx, 2.0 * math.pi * my))
    
    n_pw_total = len(G_vectors)
    
    k_points = []
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_points.append((math.pi * t, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_points.append((math.pi, math.pi * t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_points.append((math.pi * (1 - t), math.pi * (1 - t)))
    
    bands = []
    
    for kx, ky in k_points:
        M = np.zeros((n_pw_total, n_pw_total))
        for i in range(n_pw_total):
            kx_i = kx + G_vectors[i][0]
            ky_i = ky + G_vectors[i][1]
            for j in range(n_pw_total):
                kx_j = kx + G_vectors[j][0]
                ky_j = ky + G_vectors[j][1]
                G_diff_x = G_vectors[i][0] - G_vectors[j][0]
                G_diff_y = G_vectors[i][1] - G_vectors[j][1]
                mx = int(round(G_diff_x / (2.0 * math.pi)))
                my = int(round(G_diff_y / (2.0 * math.pi)))
                f_val = inv_eps_coeff.get((mx, my), 0.0)
                M[i, j] = (kx_i * kx_j + ky_i * ky_j) * f_val.real
        
        eigenvalues = np.linalg.eigvalsh(M)
        omegas = [math.sqrt(max(0.0, ev)) for ev in eigenvalues]
        bands.append((kx, ky, omegas))
    
    return bands


def _compute_gap_ratio(bands, lower_band=4):
    """Compute the gap-midgap ratio between bands 4 and 5."""
    max_omega_lower = 0.0
    min_omega_upper = float('inf')
    
    for kx, ky, omegas in bands:
        if len(omegas) > lower_band:
            max_omega_lower = max(max_omega_lower, omegas[lower_band - 1])
            min_omega_upper = min(min_omega_upper, omegas[lower_band])
    
    if min_omega_upper == float('inf'):
        return -1.0
    
    gap = min_omega_upper - max_omega_lower
    midgap = (min_omega_upper + max_omega_lower) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap


def _four_rod_pattern(size, r1, r2, r3, r4, offset):
    """Create a four-rod pattern at (±offset, ±offset) positions."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    
    centers = [
        (offset, offset),
        (-offset, offset),
        (offset, -offset),
        (-offset, -offset),
    ]
    radii = [r1, r2, r3, r4]
    
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            for c, r in zip(centers, radii):
                dx = x - c[0]
                dy = y - c[1]
                if dx * dx + dy * dy < r * r:
                    quarter[i][j] = 1
                    break
    return quarter


def _single_rod_pattern(size, radius):
    """Create a single centered rod pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                quarter[i][j] = 1
    return quarter


def design(problem):
    """Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210.0  # Leave 20s margin
    
    # Start with the four-rod pattern that scored well
    best_cell = None
    best_score = -float('inf')
    
    def evaluate_cell(cell):
        """Evaluate the gap ratio for a given cell."""
        try:
            n_pw = 17
            inv_eps_coeff = _compute_1over_eps_fourier(cell, size, n_pw)
            bands = _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=30)
            score = _compute_gap_ratio(bands, lower_band=4)
            return score
        except Exception:
            return -float('inf')
    
    # Initial candidate: four rods at (±1/4, ±1/4) with radius ~0.15
    initial_offset = 0.25
    initial_radius = 0.15
    
    quarter = _four_rod_pattern(size, initial_radius, initial_radius, initial_radius, initial_radius, initial_offset)
    cell = _make_symmetric_cell(size, quarter)
    score = evaluate_cell(cell)
    if score > best_score:
        best_score = score
        best_cell = cell
    
    # Use scipy.optimize to find better parameters
    # We'll optimize offset and radius
    def objective(params):
        offset = params[0]
        radius = params[1]
        
        # Check time
        if time.time() > deadline - 10.0:
            return -100.0
        
        quarter = _four_rod_pattern(size, radius, radius, radius, radius, offset)
        cell = _make_symmetric_cell(size, quarter)
        score = evaluate_cell(cell)
        
        # We want to maximize, so return negative
        return -score
    
    # Try multiple starting points
    start_points = [
        [0.25, 0.15],
        [0.20, 0.18],
        [0.30, 0.12],
        [0.22, 0.16],
    ]
    
    for sp in start_points:
        if time.time() > deadline - 15.0:
            break
        
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                             options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.01})
            
            if result.fun < -best_score:
                offset, radius = result.x
                quarter = _four_rod_pattern(size, radius, radius, radius, radius, offset)
                cell = _make_symmetric_cell(size, quarter)
                score = evaluate_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
        except Exception:
            pass
    
    # Try single rod for comparison
    for radius in [0.15, 0.18, 0.20, 0.22]:
        if time.time() > deadline - 10.0:
            break
        quarter = _single_rod_pattern(size, radius)
        cell = _make_symmetric_cell(size, quarter)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    if best_cell is None:
        best_cell = _make_symmetric_cell(size, _single_rod_pattern(size, 0.2))
    
    return best_cell

# EVOLVE-BLOCK-END
