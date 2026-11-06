import numpy as np
import math

def _fourier_coeffs(cell, max_g):
    """
    Compute Fourier coefficients of the inverse dielectric constant.
    cell: 32x32 list of 0/1.
    max_g: maximum G index.
    Returns a dict mapping (gx, gy) to complex coefficient of 1/epsilon.
    """
    size = len(cell)
    eps = np.array(cell, dtype=np.float64)
    inv_eps = np.where(eps == 1, 1.0, 1.0/12.0)
    
    coeffs = {}
    for gx in range(-max_g, max_g + 1):
        for gy in range(-max_g, max_g + 1):
            phase_x = np.exp(-1j * 2 * np.pi * gx * np.arange(size) / size)
            phase_y = np.exp(-1j * 2 * np.pi * gy * np.arange(size) / size)
            phases = np.outer(phase_x, phase_y)
            c = np.sum(inv_eps * phases)
            coeffs[(gx, gy)] = c / (size * size)
            
    return coeffs

def _solve_tm_bands(coeffs, kx, ky, max_g):
    """
    Solve for TM bands at given kx, ky (in units of 2*pi/a).
    Returns sorted eigenvalues.
    """
    g_vals = list(range(-max_g, max_g + 1))
    g_vecs = [(m, n) for m in g_vals for n in g_vals]
    n_pw = len(g_vecs)
    
    matrix = np.zeros((n_pw, n_pw), dtype=np.float64)
    
    for i, (mi, ni) in enumerate(g_vecs):
        for j, (mj, nj) in enumerate(g_vecs):
            kx_i = kx + mi
            ky_i = ky + ni
            kx_j = kx + mj
            ky_j = ky + nj
            
            dot_prod = kx_i * kx_j + ky_i * ky_j
            
            dgx = mi - mj
            dgy = ni - nj
            
            if abs(dgx) <= max_g and abs(dgy) <= max_g:
                eps_coeff = coeffs.get((dgx, dgy), 0)
            else:
                eps_coeff = 0
            
            if eps_coeff != 0:
                matrix[i, j] = dot_prod / eps_coeff
                
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.sort(eigvals)
    return eigvals

def design(problem):
    size = problem["pixels"]
    
    def random_symmetric_cell():
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if np.random.rand() > 0.5:
                    # Flip orbit
                    points = [
                        (i, j),
                        (j, size-1-i),
                        (size-1-i, size-1-j),
                        (size-1-j, i),
                        (size-1-i, j),
                        (i, size-1-j),
                        (j, i),
                        (size-1-j, size-1-i)
                    ]
                    for pi, pj in points:
                        cell[pi][pj] = 1
        return cell

    max_g = 2
    
    k_points = []
    for t in [0.0, 0.25, 0.5]:
        k_points.append((t, 0.0))
        k_points.append((0.5, t))
        k_points.append((0.5 - t, 0.5 - t))
    k_points = list(set(k_points))
    
    def evaluate(cell):
        coeffs = _fourier_coeffs(cell, max_g)
        min_omega5 = float('inf')
        max_omega4 = 0
        
        for kx, ky in k_points:
            eigvals = _solve_tm_bands(coeffs, kx, ky, max_g)
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w4 > max_omega4:
                    max_omega4 = w4
                if w5 < min_omega5:
                    min_omega5 = w5
                    
        if min_omega5 <= max_omega4:
            return -1.0
            
        gap = min_omega5 - max_omega4
        mid = (min_omega5 + max_omega4) / 2
        if mid == 0:
            return 0.0
        return gap / mid

    current_cell = random_symmetric_cell()
    current_score = evaluate(current_cell)
    best_cell = [row[:] for row in current_cell]
    best_score = current_score
    
    T = 1.0
    cooling = 0.99
    
    for it in range(1500):
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        orbit = set()
        points = [
            (i, j),
            (j, size-1-i),
            (size-1-i, size-1-j),
            (size-1-j, i),
            (size-1-i, j),
            (i, size-1-j),
            (j, i),
            (size-1-j, size-1-i)
        ]
        for pi, pj in points:
            orbit.add((pi, pj))
            
        new_cell = [row[:] for row in current_cell]
        for pi, pj in orbit:
            new_cell[pi][pj] = 1 - new_cell[pi][pj]
            
        new_score = evaluate(new_cell)
        
        if new_score > current_score or np.random.rand() < math.exp((new_score - current_score) / T):
            current_cell = new_cell
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
                
        T *= cooling
        
    return best_cell
