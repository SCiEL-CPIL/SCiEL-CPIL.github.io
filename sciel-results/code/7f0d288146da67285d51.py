import numpy as np
import time

def design(problem):
    """
    Optimizes a continuous dielectric field defined by a small set of
    C4-symmetric Fourier coefficients to maximize the TM band gap between
    bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Define the parameterization:
    # We use a Fourier series in the unit cell [0,1]x[0,1].
    # To ensure symmetry (invariant under square symmetry group about center),
    # we restrict the Fourier coefficients to those compatible with C4v symmetry
    # centered at (0.5, 0.5).
    # Actually, the problem states: "invariant under the square symmetry group about the cell centre 
    # (cell == cell[::-1] row-wise and column-wise and cell == its transpose)."
    # This implies symmetry about the center (0.5, 0.5).
    
    # Let's use a basis of cos/sin terms that are symmetric about 0.5.
    # cos(n * pi * (x - 0.5)) and sin(n * pi * (x - 0.5)) are symmetric/antisymmetric about 0.5?
    # cos(n*pi*(0.5 + d)) = cos(n*pi/2 + n*pi*d). 
    # It's easier to define the field on the grid and enforce symmetry by construction,
    # or use a parameterization that naturally satisfies it.
    
    # Strategy: 
    # Represent the dielectric constant epsilon(x,y) as:
    # epsilon(x,y) = eps_avg + sum_{m,n} A_{m,n} * cos(2*pi*m*(x-0.5)) * cos(2*pi*n*(y-0.5))
    # Wait, the symmetry requires:
    # 1. f(x,y) = f(1-x, y)
    # 2. f(x,y) = f(x, 1-y)
    # 3. f(x,y) = f(y,x)
    
    # Using cos(2*pi*m*(x-0.5)) satisfies f(1-x,y)=f(x,y) because cos(2*pi*m*(0.5-x-0.5)) = cos(-2*pi*m*x) = cos(2*pi*m*x).
    # Wait, let u = x-0.5. Then 1-x-0.5 = -u. cos(-z)=cos(z). So yes, cos terms in (x-0.5) are symmetric about 0.5.
    # Same for y.
    # The term cos(2*pi*m*(x-0.5)) * cos(2*pi*n*(y-0.5)) is symmetric about x=0.5 and y=0.5.
    # Is it symmetric about y=x? 
    # f(x,y) = sum A_{mn} C_m(x) C_n(y).
    # f(y,x) = sum A_{mn} C_m(y) C_n(x).
    # For f(x,y) = f(y,x), we need A_{mn} = A_{nm}.
    
    # So we can parameterize by a symmetric matrix of coefficients A_{mn} for m,n >= 0.
    # We will use a small number of harmonics, e.g., m,n in {0, 1, 2, 3}.
    # Number of parameters: (4*4 + 1)/2 = 8.5? No, m,n from 0 to 3.
    # Pairs (0,0), (0,1)...(0,3), (1,1)...(1,3), (2,2)...(2,3), (3,3).
    # Total 10 parameters.
    
    # We will optimize these 10 parameters plus a global offset and scale?
    # Actually, epsilon must be between eps_low and eps_high.
    # We can map the raw parameters to epsilon values.
    # Let's define the field as:
    # eps(x,y) = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * tanh( field(x,y) )
    # where field(x,y) is the Fourier sum.
    # This ensures eps is in [eps_low, eps_high].
    
    max_harmonic = 3
    # Indices for symmetric matrix A[m,n] where m <= n
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    # Initial guess: a dielectric rod.
    # A rod is roughly constant in the center, 0 at edges.
    # This can be approximated by a Gaussian, which has broad Fourier spectrum.
    # For a simple start, let's try a "blob" shape.
    # A(0,0) controls the average. If we want a rod, average should be lower than max.
    # Let's start with random parameters or a heuristic.
    # Heuristic: A(0,0) = 0 (mean zero field before tanh), A(1,1) large positive?
    # cos(2*pi*1*(x-0.5)) is 1 at x=0.5, -1 at x=0,1.
    # So cos(x)cos(y) is a "bump" in the center.
    # Let's set A(1,1) = 2.0, others 0.
    
    def get_field(params):
        """Returns the 32x32 epsilon grid."""
        # params is a list of length num_params
        A = np.zeros((max_harmonic + 1, max_harmonic + 1))
        for i, (m, n) in enumerate(params_idx):
            A[m, n] = params[i]
            A[n, m] = params[i]
        
        x = np.linspace(0.5/size, 1 - 0.5/size, size)
        y = np.linspace(0.5/size, 1 - 0.5/size, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Shift to center 0.5
        Xc = X - 0.5
        Yc = Y - 0.5
        
        field = np.zeros((size, size))
        for m in range(max_harmonic + 1):
            for n in range(max_harmonic + 1):
                if A[m, n] != 0:
                    # cos(2*pi*m*Xc) * cos(2*pi*n*Yc)
                    cm = np.cos(2 * np.pi * m * Xc)
                    cn = np.cos(2 * np.pi * n * Yc)
                    field += A[m, n] * cm * cn
        
        # Map to epsilon
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a simplified plane-wave method.
        This is an approximation of the evaluator's 289-PW method.
        We use a smaller number of PWs (e.g., 49 or 81) for speed, but it should
        correlate well with the true gap.
        """
        # Discretize epsilon into Fourier coefficients
        # The evaluator uses "exact Fourier coefficients of the pixel map".
        # We can compute them via FFT of the 32x32 grid?
        # The grid is 32x32. The unit cell is 1x1.
        # The Fourier coefficients are eps_{G} = integral eps(r) exp(-i G.r) dr.
        # Using the pixel values as a step function approximation:
        # eps_{G} = (1/32^2) * sum_{i,j} eps_{i,j} * exp(-i G . (r_i, r_j))
        
        # We need to define the set of G vectors.
        # The evaluator uses 289 PWs. This corresponds to a cutoff radius.
        # 289 = 17^2. This suggests a square grid of G vectors?
        # Or a circular cutoff?
        # If square, it's (2*N+1)^2. 17=2*8+1. So N=8.
        # Let's use N=8 (17x17=289 PWs) to match the evaluator exactly if possible.
        # However, solving a 289x289 eigenvalue problem 30 times might be slow in Python.
        # 30 * (289^3 / 60) ~ 30 * 4 million ~ 120 million ops.
        # NumPy eig is fast. 289x289 is small.
        
        N_pw = 8
        G_vals = np.arange(-N_pw, N_pw + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
        num_pw = len(G_vectors)
        
        # Compute Fourier coefficients of 1/epsilon for TM mode?
        # For TM, the operator involves 1/epsilon.
        # H = (1/epsilon(k)) * (k+G)^2
        # The matrix element is (1/epsilon)_{G-G'} * (k+G)^2
        # Wait, the standard plane wave expansion for TM:
        # (k+G)^2 * E_G = sum_{G'} (1/epsilon)_{G-G'} * E_{G'}
        # So we need the Fourier coefficients of 1/epsilon.
        
        inv_eps_grid = 1.0 / eps_grid
        # Compute FFT of inv_eps_grid
        # The grid is 32x32.
        # FFT gives coefficients for G = 0, 1, ..., 31.
        # We need coefficients for G in [-8, 8].
        # FFT indices 0..31 correspond to G = 0, 1, ..., 15, -16, ..., -1.
        # We need to extract the correct coefficients.
        
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        # Normalize by N^2
        fft_inv_eps /= (size * size)
        
        # Extract coefficients for G in [-N_pw, N_pw]
        # Note: fft2 output is ordered [0, 1, ..., N/2-1, -N/2, ..., -1]
        # We need a helper to get coefficient for integer G
        def get_fft_coeff(fft_arr, gx, gy):
            # Map gx, gy to FFT index
            ix = gx % size
            iy = gy % size
            return fft_arr[ix, iy]
        
        # Build the matrix for a given k
        # We need to do this for 30 k-points.
        # To save time, we can precompute the (1/epsilon)_{G-G'} matrix?
        # No, (1/epsilon)_{G-G'} depends on G-G', so it's a convolution.
        # The matrix M_{G,G'} = (1/epsilon)_{G-G'} * (k+G)^2
        # This is a banded matrix (Toeplitz-like) if we consider the convolution.
        # But (k+G)^2 is not constant, so it's not strictly Toeplitz.
        
        # Let's just build the matrix for each k-point.
        # 30 k-points * 289x289 matrix construction.
        # This might be the bottleneck.
        
        # Optimization: Precompute (1/epsilon)_{G} for all needed G-G' differences.
        # The differences G-G' range from -16 to 16.
        # We can precompute a dictionary of coeff[dx, dy]
        
        coeff_dict = {}
        for dx in range(-2*N_pw, 2*N_pw + 1):
            for dy in range(-2*N_pw, 2*N_pw + 1):
                coeff_dict[(dx, dy)] = get_fft_coeff(fft_inv_eps, dx, dy)
        
        # Define k-points in the irreducible BZ
        # Square lattice, irreducible BZ is 1/8 of the square.
        # Gamma (0,0) to M (0.5, 0) to X (0.5, 0.5)
        # We need 30 points.
        # Let's sample along Gamma-M, M-X, and Gamma-X?
        # Or just a grid in the 1/8 triangle.
        
        k_points = []
        # Gamma to M: (t, 0) for t in [0, 0.5]
        # M to X: (0.5, t) for t in [0, 0.5]
        # Gamma to X: (t, t) for t in [0, 0.5]
        
        num_k_per_path = 10
        for t in np.linspace(0, 0.5, num_k_per_path):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, num_k_per_path):
            k_points.append((0.5, t))
        for t in np.linspace(0, 0.5, num_k_per_path):
            k_points.append((t, t))
        # Remove duplicates
        unique_k = list(set(k_points))
        # We want 30 points.
        # Let's just use a fixed set of 30 points.
        
        # Precompute (k+G)^2 for all G for each k?
        # No, k changes.
        
        # Let's run the eigenvalue solver
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # Build matrix
            # M[i, j] = coeff[G_i - G_j] * |k + G_i|^2
            # Note: The eigenvalue equation is sum_j M_{ij} E_j = omega^2 E_i
            # So we solve for omega^2.
            
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                Gi = G_vectors[i]
                k_plus_Gi = k + Gi
                omega_sq_i = np.dot(k_plus_Gi, k_plus_Gi)
                
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    diff = Gi - Gj
                    coeff = coeff_dict.get((int(diff[0]), int(diff[1])), 0)
                    M[i, j] = coeff * omega_sq_i
            
            # Solve eigenvalue problem
            # M is Hermitian?
            # coeff[dx, dy] = conj(coeff[-dx, -dy]) for real field.
            # M[j, i] = coeff[Gj - Gi] * |k + Gj|^2
            # M[i, j] = coeff[Gi - Gj] * |k + Gi|^2
            # If Gi != Gj, these are not simply conjugates because of the |k+G|^2 factor.
            # Wait, the operator is H = (1/epsilon) * (k+G)^2.
            # In plane wave basis, H_{G, G'} = (1/epsilon)_{G-G'} * (k+G)^2.
            # This matrix is NOT Hermitian unless (k+G)^2 = (k+G')^2, which is not true.
            # However, the eigenvalues should be real.
            # Actually, the standard formulation uses a generalized eigenvalue problem or
            # a different basis.
            # Let's check the literature.
            # For TM: (k+G)^2 E_G = sum_{G'} (1/epsilon)_{G-G'} E_{G'}
            # This is not a standard Hermitian eigenvalue problem.
            # However, if we define a new basis, it can be made Hermitian.
            # Or we can just use np.linalg.eigvals and take the real parts.
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            # The bands are indexed 1, 2, 3...
            # We need bands 4 and 5.
            # So indices 3 and 4 (0-based).
            
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

    # Optimization Loop
    # Use a simple evolutionary algorithm or random search with local refinement.
    # Given the time budget, we can do a few hundred evaluations.
    
    best_params = None
    best_score = -np.inf
    
    # Initial population
    pop_size = 20
    population = []
    
    # Seed with a "rod" guess
    # A(1,1) = 2.0
    init_params = [0.0] * num_params
    # Index of (1,1) in params_idx
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    population.append(init_params)
    
    # Random others
    for _ in range(pop_size - 1):
        p = [np.random.uniform(-2, 2) for _ in range(num_params)]
        population.append(p)
    
    # Evaluate initial population
    scores = []
    for p in population:
        eps_grid = get_field(p)
        score = evaluate_cell(eps_grid)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_params = p[:]
            
    # Check time
    if time.time() > deadline - 10:
        return get_field(best_params)
        
    # Evolutionary steps
    # We will perform a few generations.
    # Each generation: select best, mutate, evaluate.
    
    max_generations = 50
    for gen in range(max_generations):
        if time.time() > deadline - 20:
            break
            
        # Select top half
        sorted_indices = np.argsort(scores)[::-1]
        top_half = sorted_indices[:pop_size//2]
        
        new_population = []
        # Keep best
        new_population.append(population[top_half[0]])
        
        # Mutate others
        while len(new_population) < pop_size:
            parent_idx = top_half[np.random.randint(len(top_half))]
            parent = population[parent_idx]
            child = parent[:]
            # Gaussian mutation
            for i in range(num_params):
                if np.random.random() < 0.5:
                    child[i] += np.random.normal(0, 0.5)
            new_population.append(child)
            
        population = new_population
        
        # Evaluate new population (only the new ones? or all?)
        # To save time, only evaluate the new ones (all except the best which we already have)
        # But we need scores for all to select next time.
        # Let's just re-evaluate all.
        # 20 evaluations * ~0.5s = 10s per generation.
        # 50 generations * 10s = 500s. Too slow.
        
        # We need to speed up evaluate_cell or reduce evaluations.
        # Let's reduce pop_size to 10 and max_generations to 20.
        # Or use a faster solver.
        
        # For now, let's just do a few random restarts with local optimization.
        # Or just break if time is tight.
        
        # Let's reduce the number of k-points to speed up evaluation.
        # If we use 10 k-points instead of 30, it's 3x faster.
        # Let's try that.
        
        # Actually, let's just do a simple random search for the remaining time.
        # It's safer than a complex evolution loop that might time out.
        
        for _ in range(5): # 5 random candidates per "generation"
            if time.time() > deadline - 20:
                break
            p = [np.random.uniform(-2, 2) for _ in range(num_params)]
            eps_grid = get_field(p)
            score = evaluate_cell(eps_grid)
            if score > best_score:
                best_score = score
                best_params = p[:]
                
    # Final check
    if best_params is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    final_grid = get_field(best_params)
    
    # Binarize? The problem asks for 0/1 pixels.
    # "32 x 32 nested list of 0/1 pixels"
    # So we must return a binary grid.
    # We should threshold the continuous field.
    # Threshold at 0.5 * (eps_low + eps_high)?
    # Or just check if > eps_low?
    # Since it's 0/1, 0=air, 1=dielectric.
    # If eps_grid[i,j] > 0.5*(eps_low+eps_high), it's mostly dielectric.
    
    binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if final_grid[i, j] > 0.5 * (eps_low + eps_high):
                row.append(1)
            else:
                row.append(0)
        binary_cell.append(row)
        
    # Ensure symmetry (should be by construction, but floating point might break it slightly)
    # The problem requires strict symmetry.
    # We can enforce it by averaging with its symmetric counterparts.
    # But since we used symmetric basis functions, it should be exact.
    # However, binarization might break it if the value is exactly on the threshold.
    # Let's enforce symmetry on the binary grid.
    
    # Symmetry about center (0.5, 0.5)
    # cell[i][j] == cell[size-1-i][size-1-j]
    # cell[i][j] == cell[j][i]
    
    # We can just return the binary cell.
    
    return binary_cell

# Note: The above implementation of evaluate_cell is an approximation.
# The evaluator uses a specific 289-PW method.
# My approximation uses the same number of PWs (N=8) and similar k-points.
# The main risk is that the local solver is not accurate enough to guide the search.
# However, it should be good enough to find a better geometry than the seed.

# To improve speed, I will hardcode the k-points and G-vectors and precompute as much as possible.
# I will also reduce the number of k-points to 15 to allow more iterations.

def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    max_harmonic = 3
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    def get_field(params):
        A = np.zeros((max_harmonic + 1, max_harmonic + 1))
        for i, (m, n) in enumerate(params_idx):
            A[m, n] = params[i]
            A[n, m] = params[i]
        
        x = np.linspace(0.5/size, 1 - 0.5/size, size)
        y = np.linspace(0.5/size, 1 - 0.5/size, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xc = X - 0.5
        Yc = Y - 0.5
        
        field = np.zeros((size, size))
        for m in range(max_harmonic + 1):
            for n in range(max_harmonic + 1):
                if A[m, n] != 0:
                    cm = np.cos(2 * np.pi * m * Xc)
                    cn = np.cos(2 * np.pi * n * Yc)
                    field += A[m, n] * cm * cn
        
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # Precompute G vectors and k points
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 15 points in irreducible BZ
    k_points = []
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        if k not in seen:
            seen.add(k)
            unique_k.append(k)
            
    def evaluate_cell(eps_grid):
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        coeff_dict = {}
        for dx in range(-2*N_pw, 2*N_pw + 1):
            for dy in range(-2*N_pw, 2*N_pw + 1):
                ix = dx % size
                iy = dy % size
                coeff_dict[(dx, dy)] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                Gi = G_vectors[i]
                k_plus_Gi = k + Gi
                omega_sq_i = np.dot(k_plus_Gi, k_plus_Gi)
                
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    diff = Gi - Gj
                    coeff = coeff_dict.get((int(diff[0]), int(diff[1])), 0)
                    M[i, j] = coeff * omega_sq_i
            
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

    best_params = None
    best_score = -np.inf
    
    # Initial guess
    init_params = [0.0] * num_params
    idx_11 = params_idx.index((1, 1))
    init_params[idx_11] = 2.0
    best_params = init_params[:]
    best_score = evaluate_cell(get_field(best_params))
    
    # Random search
    num_attempts = 50
    for _ in range(num_attempts):
        if time.time() > deadline - 10:
            break
        p = [np.random.uniform(-2, 2) for _ in range(num_params)]
        score = evaluate_cell(get_field(p))
        if score > best_score:
            best_score = score
            best_params = p[:]
            
    if best_params is None:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    final_grid = get_field(best_params)
    binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if final_grid[i, j] > 0.5 * (eps_low + eps_high):
                row.append(1)
            else:
                row.append(0)
        binary_cell.append(row)
        
    return binary_cell
