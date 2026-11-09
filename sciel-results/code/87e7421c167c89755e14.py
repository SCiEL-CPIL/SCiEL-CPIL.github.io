import numpy as np
import time

def design(problem):
    """
    Optimizes a continuous dielectric field defined by a small set of
    C4-symmetric Fourier coefficients to maximize the TM band gap between
    bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Parameterization: C4v symmetric Fourier series centered at (0.5, 0.5)
    # f(x,y) = sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry about x=0.5, y=0.5 is automatic.
    # Symmetry about y=x requires A_{mn} = A_{nm}.
    
    max_harmonic = 3
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    # Precompute grid coordinates
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    # Precompute basis functions
    basis_funcs = []
    for (m, n) in params_idx:
        if m == 0 and n == 0:
            basis_funcs.append(np.ones((size, size)))
        else:
            cm = np.cos(2 * np.pi * m * Xc)
            cn = np.cos(2 * np.pi * n * Yc)
            basis_funcs.append(cm * cn)
    basis_funcs = np.array(basis_funcs)  # shape (num_params, size, size)
    
    def get_field(params):
        """Returns the 32x32 epsilon grid from parameters."""
        field = np.tensordot(params, basis_funcs, axes=1)
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid
    
    # Precompute G vectors and k points for the evaluator approximation
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 30 points in irreducible BZ matching the evaluator
    # Irreducible BZ of square lattice: triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # Boundary: Gamma-M: (t,0) for t in [0,0.5], M-X: (0.5,t) for t in [0,0.5], Gamma-X: (t,t) for t in [0,0.5]
    # Interior points to sample
    k_points = []
    # Gamma-M edge: 7 points
    for t in np.linspace(0, 0.5, 7):
        k_points.append((t, 0))
    # M-X edge: 7 points (excluding M which is already counted)
    for t in np.linspace(0.5, 0, 7)[1:]:
        k_points.append((0.5, t))
    # Gamma-X edge: 7 points (excluding Gamma which is already counted)
    for t in np.linspace(0, 0.5, 7)[1:]:
        k_points.append((t, t))
    # Interior points: 9 points in a grid within the triangle
    for i in range(3):
        for j in range(3):
            if i + j <= 2:
                tx = 0.5 * i / 2
                ty = 0.5 * j / 2
                if tx + ty <= 0.5:
                    k_points.append((tx, ty))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G-G' differences for each pair (i,j)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a simplified plane-wave method.
        Uses 289 PWs and ~30 k-points.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Precompute coefficient matrix for all G-G' differences
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
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
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
    
    # Optimization: Random search with adaptive local refinement
    best_params = None
    best_score = -np.inf
    
    # Initial guess: a dielectric rod-like shape
    init_params = np.zeros(num_params)
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    init_params[0] = -0.5
    
    best_params = init_params.copy()
    best_score = evaluate_cell(get_field(best_params))
    
    # Phase 1: Broad random search
    num_broad = 60
    for attempt in range(num_broad):
        if time.time() > deadline - 30:
            break
        
        if attempt < 20:
            # Local perturbation of best
            step = 0.4
            p = best_params + np.random.normal(0, step, num_params)
        else:
            # Random restart with rod bias
            p = np.random.uniform(-2, 2, num_params)
            idx = params_idx.index((1, 1))
            p[idx] = abs(p[idx]) + 1.0
            p[0] = -1.0 + np.random.uniform(-0.5, 0.5)
        
        score = evaluate_cell(get_field(p))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
    
    # Phase 2: Fine-tuning with adaptive step
    step_size = 0.15
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() < deadline - 10 and no_improve_count < max_no_improve:
        # Try multiple perturbations at current step size
        improved = False
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            p = best_params + np.random.normal(0, step_size, num_params)
            score = evaluate_cell(get_field(p))
            
            if score > best_score:
                best_score = score
                best_params = p.copy()
                improved = True
        
        if improved:
            no_improve_count = 0
            step_size *= 0.85  # Slowly reduce step
        else:
            no_improve_count += 1
            if no_improve_count % 3 == 0:
                step_size *= 0.7  # Reduce step faster if stuck
    
    # Phase 3: Final local refinement
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        p = best_params + np.random.normal(0, 0.05, num_params)
        score = evaluate_cell(get_field(p))
        if score > best_score:
            best_score = score
            best_params = p.copy()
    
    if best_params is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x_c = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y_c = (j + 0.5) / size - 0.5
                if x_c * x_c + y_c * y_c < 0.2 ** 2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
        
    final_grid = get_field(best_params)
    
    # Binarize
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = np.zeros((size, size), dtype=int)
    binary_cell[final_grid > threshold] = 1
    
    # Enforce strict symmetry by averaging with all 8 symmetric counterparts
    sym_cell = np.zeros((size, size), dtype=float)
    
    for i in range(size):
        for j in range(size):
            val = float(binary_cell[i, j])
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            for pi, pj in positions:
                sym_cell[pi, pj] += val
    
    # Threshold at 0.5 to get final binary
    final_binary = (sym_cell >= 0.5).astype(int)
    
    # Convert to nested list
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_binary[i, j]))
        final_cell.append(row)
        
    return final_cell
