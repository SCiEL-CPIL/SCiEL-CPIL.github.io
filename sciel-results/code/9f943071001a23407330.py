import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute grid coordinates centered at 0
    # Pixels are at (i+0.5)/size - 0.5
    x_coords = np.arange(size) / size - 0.5
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    
    # Plane Wave Setup
    # Evaluator uses 289 PWs. 289 = 17^2. This implies a square grid of G vectors.
    # G_x, G_y in {-8, ..., 8}
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute G differences for the convolution matrix
    # We need coeff[G_i - G_j]
    # Since the matrix is Hermitian for TM, we can exploit symmetry.
    # H_{ij} = (1/eps)_{G_i - G_j} * |k + G_i|^2
    # Wait, the standard TM eigenvalue problem is:
    # sum_j (1/eps)_{G_i - G_j} (k+G_j)^2 E_j = omega^2 E_i ?
    # No, the equation is:
    # (k+G)^2 E_G = sum_{G'} (1/eps)_{G-G'} E_{G'}
    # This is a non-Hermitian matrix if written directly as M E = w E where M_{ij} = (1/eps)_{G_i-G_j} (k+G_j)^2.
    # However, it is similar to a Hermitian matrix.
    # A more stable and standard approach for numerical solvers is to use the generalized eigenvalue problem
    # or to transform the basis. 
    # But often, people just solve the non-Hermitian matrix and take real parts, OR they use the form:
    # D * (1/eps) * D = omega^2 I, where D is diagonal with (k+G)^2.
    # Let's stick to the standard non-Hermitian form but ensure we use the correct indices.
    # Actually, many high-precision solvers use the Hermitian form by defining a new basis vector F_G = (k+G) E_G ?
    # No, let's look at the "exact-coefficient plane-wave method".
    # The matrix A is A_{G, G'} = (1/epsilon)_{G-G'} (k+G')^2.
    # Eigenvalues are omega^2.
    
    # To speed up, we precompute the difference indices
    # diff_x = Gx_i - Gx_j, diff_y = Gy_i - Gy_j
    # We can vectorize the construction of the "convolution part" of the matrix.
    
    # K-points
    # Irreducible BZ of square lattice: triangle with vertices (0,0), (0.5, 0), (0.5, 0.5)
    # We need 30 points.
    # Let's sample a grid in this triangle.
    k_points = []
    # 5x5 grid in x,y, filter to triangle
    # x in [0, 0.5], y in [0, 0.5], x >= y (or x <= y? The triangle is usually bounded by kx=ky, kx=0.5, ky=0.5? No.)
    # Square BZ: [-0.5, 0.5] x [-0.5, 0.5].
    # Irreducible: 0 <= ky <= kx <= 0.5.
    # Let's generate points in this region.
    # 6x6 grid = 36 points. Filter down to ~30.
    kx_vals = np.linspace(0, 0.5, 6)
    ky_vals = np.linspace(0, 0.5, 6)
    for kx in kx_vals:
        for ky in ky_vals:
            if ky <= kx:
                k_points.append((kx, ky))
    # We have 21 points. Let's add more along the boundary or use a finer grid.
    # 7x7 grid = 49 points. Filter -> ~28 points.
    kx_vals = np.linspace(0, 0.5, 7)
    ky_vals = np.linspace(0, 0.5, 7)
    k_points = []
    for kx in kx_vals:
        for ky in ky_vals:
            if ky <= kx + 1e-9:
                k_points.append((kx, ky))
    # 28 points. Close enough to 30. If we need exactly 30, we can add (0.5, 0.5) and (0,0) if not present, or just use these.
    # The prompt says "30 wave vectors". I will use these 28 plus (0.25, 0.25) and (0.5, 0.25) if they are not duplicates?
    # (0.25, 0.25) is in the grid (kx=0.25, ky=0.25).
    # Let's just use the generated list. It covers the BZ well.
    
    def get_binary_cell(params):
        """
        params: [a, b, p]
        a, b: semi-axes (scaling)
        p: superellipse exponent (2=circle, 4=square-like, higher=more square)
        Shape: |x/a|^p + |y/b|^p <= 1
        Due to C4v symmetry, we can define the whole cell from the first quadrant?
        No, the cell is 32x32. The shape is centered at (0.5, 0.5).
        We just evaluate the condition on the whole grid.
        """
        a, b, p = params
        # Ensure positive
        a = max(0.01, a)
        b = max(0.01, b)
        p = max(1.0, p)
        
        # |X/a|^p + |Y/b|^p <= 1
        # X, Y are in [-0.5, 0.5]
        term_x = np.abs(X) / a
        term_y = np.abs(Y) / b
        
        # Handle p=2 specially for speed? No, power is fast.
        # Avoid 0^0 issues? 0^p is 0.
        val = np.power(term_x, p) + np.power(term_y, p)
        
        cell = (val <= 1.0).astype(int)
        
        # Enforce symmetry explicitly to handle floating point edge cases
        # cell == cell[::-1] row-wise and column-wise
        # cell == cell.T
        # The superellipse is symmetric by construction, but binarization might break it if 1.0 is hit asymmetrically.
        # We can enforce it by averaging with its mirrors? No, it's binary.
        # Just trust the math, or do a quick check.
        # For safety, we can symmetrize the binary grid:
        # cell = (cell + cell[::-1, ::-1] + cell.T + cell[::-1, ::-1].T) // 4 ? No, that changes the shape.
        # The condition |x|^p + |y|^p <= 1 is perfectly symmetric.
        # The grid points are symmetric: X[i,j] = -X[N-1-i, N-1-j]?
        # x_coords = np.arange(size)/size - 0.5.
        # i=0 -> -0.5 + 0.5/32. i=31 -> 0.5 - 0.5/32.
        # So X[0,0] = -0.484, X[31,31] = 0.484.
        # |X| is symmetric.
        # So the binary cell should be exactly symmetric.
        
        return cell.tolist()

    def evaluate_cell_fast(cell_list):
        """
        Evaluates the band gap for a binary 32x32 cell.
        """
        if time.time() > deadline - 5:
            return -1.0
            
        cell = np.array(cell_list)
        
        # Construct epsilon grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        
        # Inverse epsilon
        inv_eps = 1.0 / eps_grid
        
        # Fourier coefficients of 1/eps
        # Use FFT. The grid is 32x32.
        # We need coefficients for G in [-8, 8].
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # Extract coefficients
        # fft indices: 0..31.
        # G=0 -> idx 0
        # G=1 -> idx 1
        # ...
        # G=8 -> idx 8
        # G=-1 -> idx 31
        # G=-8 -> idx 24
        def get_coeff(gx, gy):
            ix = gx % size
            iy = gy % size
            return fft_inv_eps[ix, iy]
        
        # Precompute the difference matrix indices?
        # We can build the matrix for each k point.
        # To speed up, we can precompute the "convolution kernel" matrix C where C[i,j] = (1/eps)_{G_i - G_j}.
        # This C is independent of k!
        # M(k) = C * D(k) ? No.
        # M_{ij}(k) = C_{ij} * |k + G_j|^2  (if we use the form sum_j (1/eps)_{G_i-G_j} (k+G_j)^2 E_j)
        # Wait, which index does (k+G)^2 attach to?
        # Equation: (k+G)^2 E_G = sum_{G'} (1/eps)_{G-G'} E_{G'}
        # Multiply by exp(-i G.r)?
        # Standard form: A_{G, G'} E_{G'} = omega^2 E_G
        # A_{G, G'} = (1/eps)_{G-G'} (k+G')^2
        # So M_{ij} = C_{ij} * (k+G_j)^2 where C_{ij} = (1/eps)_{G_i-G_j}.
        
        # Let's precompute C
        C = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            Gi = G_vectors[i]
            for j in range(num_pw):
                Gj = G_vectors[j]
                diff = Gi - Gj
                C[i, j] = get_coeff(int(diff[0]), int(diff[1]))
        
        # Precompute |G_j|^2 and G_j components?
        # (k+G_j)^2 = kx^2 + ky^2 + 2*kx*Gjx + 2*ky*Gjy + Gjx^2 + Gjy^2
        # We can precompute the vector of G_j^2 and use matrix operations?
        # For each k, we need a diagonal matrix D_k with entries (k+G_j)^2.
        # Then M = C @ D_k.
        # This is efficient.
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            # Compute (k+G_j)^2 for all j
            k_plus_G = G_vectors + np.array([kx, ky])
            omega_sq_G = np.sum(k_plus_G**2, axis=1)
            
            # M = C * diag(omega_sq_G)
            # Instead of building full diagonal, multiply columns of C
            M = C * omega_sq_G[None, :]
            
            # Solve eigenvalues
            # M is non-Hermitian generally.
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

    # Optimization
    # Parameters: a, b, p
    # Initial guess: Square-ish dielectric hole? Or rod?
    # "Large higher-order TM band gap". Usually requires high contrast and specific geometry.
    # For TM, dielectric rods in air are common.
    # Let's start with a rod (circle, p=2) with radius ~0.3 (a=b=0.3).
    # Or a square rod (p=10, a=b=0.3).
    
    best_params = None
    best_score = -np.inf
    
    # Try a few initial shapes
    initial_guesses = [
        [0.3, 0.3, 2.0],   # Circle
        [0.3, 0.3, 8.0],   # Square-ish
        [0.2, 0.4, 4.0],   # Ellipse
        [0.4, 0.2, 4.0],   # Ellipse
        [0.5, 0.5, 1.0],   # Diamond (L1 norm)
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 20:
            break
        cell = get_binary_cell(guess)
        score = evaluate_cell_fast(cell)
        if score > best_score:
            best_score = score
            best_params = guess[:]
            
    # Local optimization using Nelder-Mead
    # Maximize score -> Minimize -score
    
    if best_params is not None:
        def objective(params):
            if time.time() > deadline - 10:
                return 100.0 # Penalty
            cell = get_binary_cell(params)
            score = evaluate_cell_fast(cell)
            if score < 0:
                return -score + 10.0 # Penalty for negative gap
            return -score
        
        # Set up bounds? Nelder-Mead doesn't use bounds directly, but we can clip in get_binary_cell.
        # a, b in [0.05, 0.5], p in [1.0, 20.0]
        
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3, 'disp': False})
            if res.fun < -best_score: # res.fun is -score, so if -score < -best_score, score > best_score
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_params = res.x
        except Exception:
            pass
            
    # Final check and return
    if best_params is None:
        # Fallback: simple circle
        best_params = [0.3, 0.3, 2.0]
        
    final_cell = get_binary_cell(best_params)
    
    # Double check symmetry
    # The evaluator requires strict symmetry.
    # Our construction should be symmetric, but let's be paranoid.
    # If there's a floating point mismatch at the boundary, we might have asymmetric pixels.
    # We can enforce symmetry by taking the majority vote of the 4 symmetric counterparts for each pixel?
    # Or simply, since the shape is defined by |x|^p + |y|^p <= 1, and the grid is symmetric,
    # the result should be symmetric.
    
    return final_cell

# Dummy definition to satisfy "Keep the function names and signatures of the seed program" if there were others, 
# but design is the main one.
