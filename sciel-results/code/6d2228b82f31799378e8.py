import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell with four air holes in a
    dielectric background, using superellipse shapes. Optimizes hole
    parameters (radius, exponent, position) to maximize the TM band gap
    between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s budget
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute G vectors for plane-wave method
    N_pw = 17  # 17x17 = 289 plane waves
    G_vals = np.arange(-N_pw // 2, N_pw // 2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 30 points in irreducible BZ
    # Irreducible BZ of square lattice: 0 <= ky <= kx <= 0.5
    k_points = []
    # Gamma-M line: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0))
    # M-X line: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Gamma-X line: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    # Interior points
    for t1 in np.linspace(0.1, 0.4, 3):
        for t2 in np.linspace(0.1, t1, 3):
            k_points.append((t1, t2))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G-G' differences
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    def generate_cell(params):
        """
        Generate a 32x32 cell from parameters.
        params: [r, n, dx, dy] where r=radius, n=superellipse exponent,
                dx,dy = offset of hole centers from quadrant center
        """
        r, n, dx, dy = params
        
        # Create dielectric background (1 = dielectric)
        cell = np.ones((size, size), dtype=int)
        
        # Four holes at positions related by C4v symmetry
        # Quadrant centers are at (8,8), (23,8), (8,23), (23,23) in a 32x32 grid
        # Actually, for a 32x32 cell with symmetry about center (15.5, 15.5):
        # Hole positions in first quadrant: (8+dx, 8+dy)
        # By symmetry, other holes are at (23-dx, 8+dy), (8+dx, 23-dy), (23-dx, 23-dy)
        
        # Define superellipse: |x/a|^n + |y/b|^n <= 1
        # Here a = b = r (circular when n=2)
        
        x_coords = np.arange(size)
        y_coords = np.arange(size)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Four hole centers
        centers = [
            (8.0 + dx, 8.0 + dy),
            (23.0 - dx, 8.0 + dy),
            (8.0 + dx, 23.0 - dy),
            (23.0 - dx, 23.0 - dy)
        ]
        
        for cx, cy in centers:
            dx_arr = X - cx
            dy_arr = Y - cy
            # Superellipse condition
            mask = np.abs(dx_arr) / r + np.abs(dy_arr) / r <= 1.0
            # For superellipse: |x/a|^n + |y/b|^n <= 1
            if n > 0:
                mask = (np.abs(dx_arr / r) ** n + np.abs(dy_arr / r) ** n) <= 1.0
            cell[mask] = 0
        
        # Enforce strict symmetry
        cell = symmetrize(cell)
        
        return cell
    
    def symmetrize(cell):
        """Enforce C4v symmetry: cell == cell[::-1] row-wise and column-wise, cell == cell.T"""
        size = cell.shape[0]
        # Average with all 8 symmetric counterparts
        sym_cell = np.zeros_like(cell, dtype=float)
        count = np.zeros_like(cell, dtype=float)
        
        for i in range(size):
            for j in range(size):
                val = cell[i, j]
                positions = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, i),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                for pi, pj in positions:
                    sym_cell[pi, pj] += val
                    count[pi, pj] += 1
        
        # Binarize: 1 if average > 0.5
        result = (sym_cell / count) > 0.5
        return result.astype(int)
    
    def evaluate_cell(cell):
        """
        Compute the band gap using plane-wave method.
        cell: 32x32 array of 0/1 (0=air, 1=dielectric)
        Returns: gap-midgap ratio
        """
        # Create epsilon grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT of 1/epsilon
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Build coefficient matrix
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            # For TM mode, the eigenvalue problem is:
            # sum_j (k+G_i) dot (k+G_j) * (1/eps)_{G_i - G_j} * a_j = omega^2 * a_i
            # This is a Hermitian matrix, so use eigvalsh
            # Make sure M is Hermitian
            M = (M + M.conj().T) / 2
            
            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(max(0, real_eigvals[3]))
                omega_5 = np.sqrt(max(0, real_eigvals[4]))
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio
    
    # Optimization: Random search with local refinement
    best_params = None
    best_score = -np.inf
    best_cell = None
    
    # Parameter ranges:
    # r: radius (0.5 to 4.0)
    # n: superellipse exponent (1 to 10)
    # dx, dy: offset (-3 to 3)
    
    def random_params():
        r = np.random.uniform(0.5, 4.0)
        n = np.random.uniform(1.0, 10.0)
        dx = np.random.uniform(-3.0, 3.0)
        dy = np.random.uniform(-3.0, 3.0)
        return [r, n, dx, dy]
    
    # Initial guess
    init_params = [2.0, 2.0, 0.0, 0.0]  # Circular holes
    best_params = init_params.copy()
    best_cell = generate_cell(best_params)
    best_score = evaluate_cell(best_cell)
    
    # Random search
    num_attempts = 200
    for attempt in range(num_attempts):
        if time.time() > deadline - 15:
            break
        
        if attempt < 100:
            # Local perturbation
            p = best_params + np.random.normal(0, 0.3, 4)
            p[0] = np.clip(p[0], 0.5, 4.0)  # radius
            p[1] = np.clip(p[1], 1.0, 10.0)  # exponent
            p[2] = np.clip(p[2], -3.0, 3.0)  # dx
            p[3] = np.clip(p[3], -3.0, 3.0)  # dy
        else:
            # Random restart
            p = random_params()
        
        cell = generate_cell(p)
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
            best_cell = cell
            
            # Local refinement
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                p_local = best_params + np.random.normal(0, 0.1, 4)
                p_local[0] = np.clip(p_local[0], 0.5, 4.0)
                p_local[1] = np.clip(p_local[1], 1.0, 10.0)
                p_local[2] = np.clip(p_local[2], -3.0, 3.0)
                p_local[3] = np.clip(p_local[3], -3.0, 3.0)
                cell_local = generate_cell(p_local)
                score_local = evaluate_cell(cell_local)
                if score_local > best_score:
                    best_score = score_local
                    best_params = p_local.copy()
                    best_cell = cell_local
    
    if best_cell is None:
        # Fallback: simple circular holes
        cell = np.ones((size, size), dtype=int)
        for cx, cy in [(8, 8), (23, 8), (8, 23), (23, 23)]:
            for i in range(size):
                for j in range(size):
                    if (i - cx) ** 2 + (j - cy) ** 2 < 4:
                        cell[i, j] = 0
        best_cell = symmetrize(cell)
    
    # Convert to nested list
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        final_cell.append(row)
    
    return final_cell
