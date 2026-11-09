import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell optimized to maximize the TM band gap
    between bands 4 and 5 using a diverse shape search with efficient evaluation.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    def make_cell_4shape(shape_type, radius, offset, size=32):
        """Create a symmetric 4-shape cell."""
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (size * 0.25 + offset, size * 0.25 + offset),
            (size * 0.75 - offset, size * 0.25 + offset),
            (size * 0.25 + offset, size * 0.75 - offset),
            (size * 0.75 - offset, size * 0.75 - offset),
        ]
        for i in range(size):
            for j in range(size):
                y_pix = i + 0.5
                x_pix = j + 0.5
                for cx, cy in centers:
                    dx = x_pix - cx
                    dy = y_pix - cy
                    if radius <= 0:
                        continue
                    if shape_type == 'circle':
                        if dx*dx + dy*dy < radius*radius:
                            cell[i, j] = 1
                            break
                    elif shape_type == 'square':
                        if abs(dx) < radius and abs(dy) < radius:
                            cell[i, j] = 1
                            break
                    elif shape_type == 'diamond':
                        if abs(dx) + abs(dy) < radius:
                            cell[i, j] = 1
                            break
                    elif shape_type == 'superellipse':
                        val = abs(dx/radius)**4 + abs(dy/radius)**4
                        if val < 1.0:
                            cell[i, j] = 1
                            break
        return cell
    
    def make_cell_single(radius, size=32):
        """Create a single large dielectric blob at center."""
        cell = np.zeros((size, size), dtype=int)
        cx, cy = size/2.0, size/2.0
        for i in range(size):
            for j in range(size):
                dx = (j + 0.5) - cx
                dy = (i + 0.5) - cy
                if dx*dx + dy*dy < radius*radius:
                    cell[i, j] = 1
        return cell
    
    def make_cell_cross(arm_width, arm_length, size=32):
        """Create a cross/plus shape at center."""
        cell = np.zeros((size, size), dtype=int)
        cx, cy = size//2 - 0.5, size//2 - 0.5
        half_w = arm_width / 2.0
        half_l = arm_length / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs((j + 0.5) - cx)
                dy = abs((i + 0.5) - cy)
                if (dx < half_w and dy < half_l) or (dy < half_w and dx < half_l):
                    cell[i, j] = 1
        return cell
    
    def get_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon for the cell."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        return np.fft.fft2(inv_eps) / (size * size)
    
    def solve_bands(cell, n_pw=14, k_pts=None):
        """
        TM band solver using plane-wave expansion.
        H[i,j] = |k-G_i|² * ε⁻¹[G_i - G_j]
        """
        fft_2d = get_fft_inv_eps(cell)
        
        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx].real
        
        if k_pts is None:
            k_pts = [
                (0.0, 0.0),      # Γ
                (0.5, 0.0),      # X
                (0.5, 0.5),      # M
                (0.25, 0.0),     
                (0.25, 0.25),    
                (0.5, 0.25),     
                (0.1, 0.0),
                (0.1, 0.1),
                (0.3, 0.0),
                (0.3, 0.3),
            ]
        
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))
        
        n_pw_actual = len(g_pairs)
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in k_pts:
            H = np.zeros((n_pw_actual, n_pw_actual), dtype=float)
            
            for i, (mi, ni) in enumerate(g_pairs):
                kx_i = kx - mi
                ky_i = ky - ni
                k_sq_i = kx_i * kx_i + ky_i * ky_i
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    H[i, j] = k_sq_i * eps_val
            
            H = (H + H.T) / 2.0
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(eigenvalues)
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    best_score = -1e10
    best_cell = None
    
    # Phase 1: Coarse search
    shape_types = ['circle', 'square', 'diamond', 'superellipse']
    radii = [4.0, 5.0, 6.0, 7.0, 8.0]
    offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for st in shape_types:
        for r in radii:
            for off in offsets:
                if time.time() > deadline:
                    break
                cell = make_cell_4shape(st, r, off)
                score = solve_bands(cell, n_pw=12)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
        if time.time() > deadline:
            break
    
    # Try single blob and cross shapes
    for r in [8.0, 10.0, 12.0, 14.0]:
        if time.time() > deadline:
            break
        cell = make_cell_single(r)
        score = solve_bands(cell, n_pw=12)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    for aw in [2, 4, 6, 8]:
        for al in [10, 14, 18, 22]:
            if time.time() > deadline:
                break
            cell = make_cell_cross(aw, al)
            score = solve_bands(cell, n_pw=12)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Phase 2: Fine-tune the best shape type
    if best_cell is not None and time.time() < deadline - 40:
        # Determine which shape type was best and refine
        # Try nearby parameters for each shape type
        for st in shape_types:
            for r in np.arange(4.0, 9.0, 0.5):
                for off in np.arange(-2.0, 3.0, 0.5):
                    if time.time() > deadline - 15:
                        break
                    cell = make_cell_4shape(st, r, off)
                    score = solve_bands(cell, n_pw=14)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
            if time.time() > deadline - 15:
                break
    
    # Convert to nested list
    if best_cell is None:
        best_cell = make_cell_4shape('circle', 6.0, 0.0)
    
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
