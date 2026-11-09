import numpy as np
import time

def design(problem):
    """
    Optimizes a continuous dielectric field defined by a small set of
    C4-symmetric Fourier coefficients to maximize the TM band gap between
    bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 230s kill
    
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

    # --- FAST EVALUATOR FOR SEARCH ---
    # Use fewer PWs and k-points to allow more iterations.
    N_pw = 4  # 9x9 = 81 PWs (down from 289)
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Fewer k-points: 10 points covering the irreducible BZ
    k_points = []
    # Gamma-M
    k_points.append((0.0, 0.0))
    k_points.append((0.25, 0.0))
    k_points.append((0.5, 0.0))
    # M-X
    k_points.append((0.5, 0.25))
    # Gamma-X
    k_points.append((0.25, 0.25))
    # Interior
    k_points.append((0.1, 0.1))
    k_points.append((0.2, 0.2))
    k_points.append((0.3, 0.1))
    k_points.append((0.1, 0.3))
    k_points.append((0.4, 0.1))
    
    # Precompute G_diff
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    dx = G_diff[:, :, 0]
    dy = G_diff[:, :, 1]
    ix = dx % size
    iy = dy % size
    
    def evaluate_cell(eps_grid):
        """
        Fast approximation of the band gap.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Precompute coefficient matrix for this grid
        coeff_matrix = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            # Use eigh for Hermitian matrices (M is not strictly Hermitian in this simple formulation 
            # if we don't account for complex phases correctly, but for real symmetric epsilon, 
            # the operator is real symmetric. However, the plane wave matrix M_{ij} = <G_i|1/eps|G_j> * |k+G_j|^2
            # is generally complex if epsilon is not constant. Here epsilon is real, so 1/eps is real.
            # The matrix M is real. Is it symmetric? 
            # M_{ij} = C_{ij} * omega_j^2. C_{ij} is symmetric (Fourier coeff of real function).
            # So M is NOT symmetric because of the omega_j^2 factor on columns.
            # We must use eigvals, not eigh.
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Ensure positive eigenvalues for sqrt
                valid_eigs = real_eigvals[real_eigvals > 1e-12]
                if len(valid_eigs) >= 5:
                    omega_4 = np.sqrt(valid_eigs[3])
                    omega_5 = np.sqrt(valid_eigs[4])
                    
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
    
    # Initial guess: Dielectric rod (high center, low edges) or Ring (low center, high mid, low edges)
    # Let's try a Rod first (standard high-index inclusion)
    init_params = np.zeros(num_params)
    # (0,0) is DC component. Positive value -> higher epsilon everywhere.
    # (1,1) cos(2pi(x-.5))cos(2pi(y-.5)) -> positive at center, negative at edges.
    # To make a rod (high center), we want high value at center.
    # So (1,1) should be positive.
    init_params[params_idx.index((0,0))] = 1.0
    init_params[params_idx.index((1,1))] = 2.0
    
    best_params = init_params.copy()
    best_score = evaluate_cell(get_field(best_params))
    
    # Also try a Ring (low center, high ring)
    # (1,1) negative -> low center.
    ring_params = np.zeros(num_params)
    ring_params[params_idx.index((0,0))] = 0.5
    ring_params[params_idx.index((1,1))] = -2.0
    ring_score = evaluate_cell(get_field(ring_params))
    if ring_score > best_score:
        best_score = ring_score
        best_params = ring_params.copy()

    # Phase 1: Broad random search
    num_broad = 100
    for attempt in range(num_broad):
        if time.time() > deadline - 30:
            break
        
        if attempt < 30:
            # Local perturbation of best
            step = 0.5
            p = best_params + np.random.normal(0, step, num_params)
        else:
            # Random restart
            p = np.random.uniform(-2, 2, num_params)
            # Bias towards rod or ring
            if np.random.rand() > 0.5:
                p[params_idx.index((1,1))] = abs(p[params_idx.index((1,1))]) + 1.0 # Rod bias
            else:
                p[params_idx.index((1,1))] = -abs(p[params_idx.index((1,1))]) - 1.0 # Ring bias
            p[params_idx.index((0,0))] = np.random.uniform(0, 1)
        
        score = evaluate_cell(get_field(p))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
    
    # Phase 2: Fine-tuning
    step_size = 0.2
    no_improve_count = 0
    max_no_improve = 10
    current_phase2_evals = 0
    max_phase2_evals = 500 # We can do more because it's faster

    while time.time() < deadline - 10 and no_improve_count < max_no_improve and current_phase2_evals < max_phase2_evals:
        improved = False
        for _ in range(10):
            if time.time() > deadline - 10 or current_phase2_evals >= max_phase2_evals:
                break
            p = best_params + np.random.normal(0, step_size, num_params)
            score = evaluate_cell(get_field(p))
            current_phase2_evals += 1
            
            if score > best_score:
                best_score = score
                best_params = p.copy()
                improved = True
        
        if improved:
            no_improve_count = 0
            step_size *= 0.9
        else:
            no_improve_count += 1
            if no_improve_count % 2 == 0:
                step_size *= 0.8

    # Phase 3: Final local refinement
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        p = best_params + np.random.normal(0, 0.05, num_params)
        score = evaluate_cell(get_field(p))
        if score > best_score:
            best_score = score
            best_params = p.copy()

    if best_params is None:
        # Fallback
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
    binary_cell = (final_grid > threshold).astype(int)
    
    # Enforce strict C4v symmetry using NumPy (fast)
    # Symmetries:
    # 1. Original
    # 2. Flip X (left-right)
    # 3. Flip Y (up-down)
    # 4. Flip XY (180 rot)
    # 5. Transpose
    # 6. Flip X then Transpose (or Flip Y then Transpose)
    # 7. Flip Y then Transpose
    # 8. Flip XY then Transpose

    sym_cell = np.zeros((size, size), dtype=float)
    sym_cell += binary_cell
    sym_cell += binary_cell[:, ::-1]
    sym_cell += binary_cell[::-1, :]
    sym_cell += binary_cell[::-1, ::-1]
    sym_cell += binary_cell.T
    sym_cell += binary_cell.T[:, ::-1]
    sym_cell += binary_cell.T[::-1, :]
    sym_cell += binary_cell.T[::-1, ::-1]
    
    # Average and threshold
    # 8 symmetries. If a pixel is 1 in all 8, it stays 1.
    # If it's 1 in 4, it's 0.5 -> 0.
    # This ensures strict symmetry.
    final_binary = (sym_cell >= 4).astype(int)
    
    # Convert to nested list
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_binary[i, j]))
        final_cell.append(row)
        
    return final_cell
