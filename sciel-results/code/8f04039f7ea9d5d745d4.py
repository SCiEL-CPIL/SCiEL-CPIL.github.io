import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def expand_quadrant(quadrant):
        cell = [[0]*size for _ in range(size)]
        for i in range(16):
            for j in range(16):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][31-j] = val
                cell[31-i][j] = val
                cell[31-i][31-j] = val
        return cell

    def compute_fourier_coeffs_eps(cell, n_max):
        N = size
        eps_arr = np.array(cell, dtype=float)
        eps_arr[eps_arr == 1] = eps_high
        eps_arr[eps_arr == 0] = eps_low
        
        i_coords = np.arange(N) + 0.5
        j_coords = np.arange(N) + 0.5
        I, J = np.meshgrid(i_coords, j_coords, indexing='ij')
        
        coeffs = {}
        for m in range(-n_max, n_max + 1):
            for n in range(-n_max, n_max + 1):
                phase = np.exp(-1j * 2 * np.pi * (m * I / N + n * J / N))
                sx = np.sin(np.pi * m / N) / (np.pi * m / N) if m != 0 else 1.0
                sy = np.sin(np.pi * n / N) / (np.pi * n / N) if n != 0 else 1.0
                c = np.sum(eps_arr * phase) * sx * sy
                coeffs[(m,n)] = c / (N * N)
        return coeffs

    def solve_tm_bands_v2(coeffs_eps, kx, ky, n_max):
        indices = [(m, n) for m in range(-n_max, n_max + 1) for n in range(-n_max, n_max + 1)]
        N_modes = len(indices)
        
        A = np.zeros((N_modes, N_modes), dtype=complex)
        B = np.zeros((N_modes, N_modes), dtype=complex)
        
        for i, (m, n) in enumerate(indices):
            Gx = m
            Gy = n
            kx_total = kx + Gx
            ky_total = ky + Gy
            B[i, i] = kx_total * kx_total + ky_total * ky_total
            
            for j, (mp, np_) in enumerate(indices):
                dm = m - mp
                dn = n - np_
                if abs(dm) <= n_max and abs(dn) <= n_max:
                    A[i, j] = coeffs_eps.get((dm, dn), 0.0)
                else:
                    A[i, j] = 0.0
            
        try:
            eigs = np.linalg.eigvalsh(A, B)
            omegas = []
            for lam in eigs:
                if lam > 1e-9:
                    omegas.append(1.0 / np.sqrt(lam))
            omegas.sort()
            return omegas
        except Exception:
            return []

    n_max = 8
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5)
    ]
    
    def evaluate_cell(cell):
        coeffs = compute_fourier_coeffs_eps(cell, 16)
        max_gap = -1.0
        for kx, ky in k_points:
            omegas = solve_tm_bands_v2(coeffs, kx, ky, 8)
            if len(omegas) >= 5:
                w4 = omegas[3]
                w5 = omegas[4]
                gap = 2 * (w5 - w4) / (w5 + w4)
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    quadrant = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            if i >= 12 or j >= 12:
                quadrant[i][j] = 1
    
    best_cell = expand_quadrant(quadrant)
    best_score = evaluate_cell(best_cell)
    
    while time.time() < deadline:
        ri = np.random.randint(0, 16)
        rj = np.random.randint(0, 16)
        
        old_val = quadrant[ri][rj]
        quadrant[ri][rj] = 1 - old_val
        
        cell = expand_quadrant(quadrant)
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
        else:
            quadrant[ri][rj] = old_val

    return best_cell
