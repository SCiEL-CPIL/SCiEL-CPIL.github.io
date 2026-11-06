import numpy as np
import time
import sys

def _build_pwe_matrix(k, G_list, get_eps_inv):
    """
    Builds the PWE matrix for TM polarization.
    k: (kx, ky)
    G_list: list of (Gx, Gy)
    get_eps_inv: callable(gx, gy) returning epsilon_inv Fourier coeff
    """
    N = len(G_list)
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        kx_i = k[0] + G_list[i][0]
        ky_i = k[1] + G_list[i][1]
        for j in range(N):
            kx_j = k[0] + G_list[j][0]
            ky_j = k[1] + G_list[j][1]
            G_diff_x = G_list[i][0] - G_list[j][0]
            G_diff_y = G_list[i][1] - G_list[j][1]
            val = get_eps_inv(G_diff_x, G_diff_y)
            # TM operator: (k+G_i) . (k+G_j) * eps_inv(G_i - G_j)
            dot = kx_i * kx_j + ky_i * ky_j
            M[i, j] = dot * val
    return M

def _get_G_list(n_pw):
    """
    Generates G vectors for a square lattice.
    """
    M = int(np.ceil(np.sqrt(n_pw)))
    if M % 2 == 0:
        M += 1
    half = M // 2
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    return G_list

def _calculate_gap(cell, n_pw=25):
    """
    Calculates the gap-midgap ratio for TM bands 4 and 5.
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute eps_inv map
    eps_inv = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                eps_inv[i, j] = 1.0 / eps_high
            else:
                eps_inv[i, j] = 1.0 / eps_low
    
    # Compute Fourier coefficients of eps_inv
    fft_eps_inv = np.fft.fft2(eps_inv)
    
    def get_eps_inv(gx, gy):
        ix = gx % size
        iy = gy % size
        return fft_eps_inv[ix, iy] / (size * size)
        
    G_list = _get_G_list(n_pw)
    
    # k-points in the irreducible BZ
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.5, 0.25)
    ]
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        k = (kx, ky)
        M = _build_pwe_matrix(k, G_list, get_eps_inv)
        # M is Hermitian
        eigs = np.linalg.eigvalsh(M)
        omegas = np.sqrt(eigs)
        omegas.sort()
        if len(omegas) >= 5:
            w4 = omegas[3]
            w5 = omegas[4]
            if w4 > max_omega_4:
                max_omega_4 = w4
            if w5 < min_omega_5:
                min_omega_5 = w5
                
    if min_omega_5 == float('inf') or max_omega_4 == 0.0:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return -1.0
    return gap / midgap

def _make_cell_circle(radius):
    size = 32
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < radius*radius:
                cell[i][j] = 1
    return cell

def _make_cell_square(side):
    size = 32
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    half_side = side / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            if dx < half_side and dy < half_side:
                cell[i][j] = 1
    return cell

def _make_cell_2x2_circles(radius, offset=8.0):
    size = 32
    cell = [[0]*size for _ in range(size)]
    # Symmetric centers: (off, off), (31-off, off), (off, 31-off), (31-off, 31-off)
    c1 = offset
    c2 = 31 - offset
    centers = [
        (c1, c1),
        (c2, c1),
        (c1, c2),
        (c2, c2)
    ]
    for i in range(size):
        for j in range(size):
            for cx, cy in centers:
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < radius*radius:
                    cell[i][j] = 1
                    break
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin for 200s limit
    
    best_score = -1.0
    best_cell = None
    
    # Candidate 1: Single Circle
    for r_norm in np.linspace(0.1, 0.4, 10):
        if time.time() > deadline:
            break
        r_pix = r_norm * 32.0
        cell = _make_cell_circle(r_pix)
        score = _calculate_gap(cell, n_pw=25)
        if score > best_score:
            best_score = score
            best_cell = cell

    # Candidate 2: Single Square
    for s_norm in np.linspace(0.1, 0.4, 10):
        if time.time() > deadline:
            break
        s_pix = s_norm * 32.0
        cell = _make_cell_square(s_pix)
        score = _calculate_gap(cell, n_pw=25)
        if score > best_score:
            best_score = score
            best_cell = cell

    # Candidate 3: 2x2 Circles
    for r_norm in np.linspace(0.05, 0.2, 8):
        if time.time() > deadline:
            break
        r_pix = r_norm * 32.0
        cell = _make_cell_2x2_circles(r_pix, offset=8.0)
        score = _calculate_gap(cell, n_pw=25)
        if score > best_score:
            best_score = score
            best_cell = cell

    if best_cell is None:
        best_cell = _make_cell_circle(0.2 * 32.0)

    return best_cell
