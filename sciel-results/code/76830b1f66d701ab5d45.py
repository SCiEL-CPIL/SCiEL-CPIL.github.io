import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell for a 2D square-lattice photonic crystal.
    Strategy:
    1. Use a "dielectric background with air holes" topology (inverted from the current square frame).
       Specifically, a large dielectric square with a central air hole, or potentially a grid of holes.
       The knowledge base suggests inverting material assignment (dielectric bg + air holes) significantly
       improves TM gaps due to field concentration in low-index regions.
    2. Use a continuous field representation via C4v-symmetric Fourier series for optimization.
    3. Optimize the Fourier coefficients using a fast approximate plane-wave method (smaller basis)
       to maximize the band gap between bands 4 and 5.
    4. Binarize the result and enforce strict C4v symmetry.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 200s limit

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
        # params is a list/array of length num_params
        # field = sum params[i] * basis_funcs[i]
        field = np.tensordot(params, basis_funcs, axes=1)
        # Map field to epsilon:
        # We want to bias towards dielectric background (high eps) with air holes (low eps).
        # So if field is high -> eps_high (dielectric), if field is low -> eps_low (air).
        # Actually, standard tanh mapping:
        # eps = mid + amp * tanh(field)
        # If we want dielectric background, the average field should be positive so tanh is positive -> high eps.
        # Air holes would be local minima in field (negative tanh) -> low eps.
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # Fast evaluator using a subset of plane waves and k-points
    # The full evaluator uses 289 PWs (17x17) and 30 k-points.
    # We use 49 PWs (7x7) and 15 k-points for speed.
    N_pw = 3 # Half-range, so -3 to 3 -> 7 values -> 49 PWs
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 15 points in irreducible BZ (Gamma-M, M-X, Gamma-X)
    k_points = []
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
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

    # Optimization: Random search with local refinement
    best_params = None
    best_score = -np.inf
    
    # Initial guess: Dielectric background with central air hole
    # To get dielectric background, A(0,0) should be positive (large tanh).
    # To get a central hole, we need a dip in the center.
    # cos(2*pi*m*x) has max at x=0.5 (center).
    # So to create a dip (low field) at center, we need negative coefficients for low-order harmonics?
    # Let's test:
    # f(0.5, 0.5) = A00 + A11 + A12 + ... (all cos terms are 1 at center)
    # If we want low field at center, we need negative sum of coefficients.
    # But we want high field elsewhere.
    # This is tricky with just a few terms.
    
    # Alternative: Start with a known good topology.
    # Dielectric square frame (current program) is essentially:
    # 1 in a square ring, 0 in center and corners.
    # Inverted: 0 in a square ring (air ring), 1 in center and corners (dielectric).
    # This is "Dielectric background with air holes" where the holes form a square ring?
    # Or simply: Dielectric everywhere, except a central square hole.
    
    # Let's try initializing with parameters that produce a central hole in a dielectric background.
    # A00 large positive -> high eps everywhere.
    # A11 negative -> subtracts from center.
    
    init_params = np.zeros(num_params)
    init_params[0] = 2.0  # Strong dielectric background
    # To make a hole in the center, we need to reduce the field at (0.5, 0.5).
    # All cos(2*pi*m*(x-0.5)) are 1 at x=0.5.
    # So we need negative coefficients for the terms that are "broad" enough to affect the center.
    # A11 is cos(2pi(x-.5))cos(2pi(y-.5)). At center it is 1. At edges (x=0, y=0.5) it is -1*1 = -1.
    # If A11 is negative, it subtracts from center and adds to edges?
    # Let's just use random restarts and see what sticks.
    
    # Restart 1: Central hole
    p1 = np.zeros(num_params)
    p1[0] = 2.0
    p1[params_idx.index((1,1))] = -2.0
    best_params = p1.copy()
    best_score = evaluate_cell(get_field(best_params))
    
    # Restart 2: Square ring of holes (inverted frame)
    # This might be harder to represent with low-order Fourier.
    
    # Random search
    num_attempts = 200
    np.random.seed(42)
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate
        if attempt < 50:
            # Local perturbation of best
            p = best_params + np.random.normal(0, 0.3, num_params)
        else:
            # Random restart
            p = np.random.uniform(-2, 2, num_params)
            # Bias towards dielectric background (positive A00)
            p[0] = abs(p[0]) + 0.5
        
        score = evaluate_cell(get_field(p))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
            
            # Local refinement around new best
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                p_local = best_params + np.random.normal(0, 0.1, num_params)
                score_local = evaluate_cell(get_field(p_local))
                if score_local > best_score:
                    best_score = score_local
                    best_params = p_local.copy()
    
    if best_params is None:
        # Fallback to seed (simple rod)
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
    
    # Enforce strict symmetry
    # cell[i][j] == cell[size-1-i][size-1-j] (180 rotation)
    # cell[i][j] == cell[j][i] (transpose)
    # cell[i][j] == cell[size-1-j][i] (reflection)
    
    # Average with symmetric counterparts to enforce symmetry
    sym_cell = np.zeros((size, size), dtype=float)
    count = np.zeros((size, size), dtype=float)
    
    for i in range(size):
        for j in range(size):
            # Identity
            sym_cell[i, j] += binary_cell[i, j]
            count[i, j] += 1
            
            # Transpose
            sym_cell[j, i] += binary_cell[i, j]
            count[j, i] += 1
            
            # 180 Rotation
            r = size - 1 - i
            c = size - 1 - j
            sym_cell[r, c] += binary_cell[i, j]
            count[r, c] += 1
            
            # Reflection across diagonal (already covered by transpose if we iterate all i,j? No, transpose is (j,i). Reflection is (j,i) too?
            # Symmetry group C4v:
            # 1. Identity (i,j)
            # 2. 90 deg rot (j, size-1-i)
            # 3. 180 deg rot (size-1-i, size-1-j)
            # 4. 270 deg rot (size-1-j, i)
            # 5. Refl x (i, size-1-j)
            # 6. Refl y (size-1-i, j)
            # 7. Refl diag (j, i)
            # 8. Refl anti-diag (size-1-j, size-1-i)
            
            # Let's just add all 8 to be safe
            sym_cell[j, size-1-i] += binary_cell[i, j]
            count[j, size-1-i] += 1
            
            sym_cell[size-1-j, i] += binary_cell[i, j]
            count[size-1-j, i] += 1
            
            sym_cell[i, size-1-j] += binary_cell[i, j]
            count[i, size-1-j] += 1
            
            sym_cell[size-1-i, j] += binary_cell[i, j]
            count[size-1-i, j] += 1
            
            sym_cell[size-1-j, size-1-i] += binary_cell[i, j]
            count[size-1-j, size-1-i] += 1
            
    # Avoid division by zero (shouldn't happen)
    count[count == 0] = 1
    sym_cell /= count
    
    # Threshold the symmetric cell
    final_binary = (sym_cell > 0.5).astype(int)
    
    # Convert to list of lists
    cell = final_binary.tolist()
    
    return cell
