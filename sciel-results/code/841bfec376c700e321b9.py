import numpy as np
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def design(problem):
    """
    Strategy: Pixel-level iterative optimization using a fast surrogate PWE solver.
    Instead of fixing the geometry to four rods, we start with a high-fill-factor 
    four-rod configuration and then perform local search (bit-flips) on the pixel 
    grid. We use symmetry constraints to reduce the search space and a truncated 
    plane-wave expansion to quickly evaluate candidate structures.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for overhead
    
    # 1. Initialize with the known good 4-rod structure
    # This provides a strong starting point (score ~2.7)
    cell = [[0 for _ in range(size)] for _ in range(size)]
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    radius = 0.18
    
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < radius * radius:
                    cell[i][j] = 1
                    break
    
    # Convert to numpy array for speed
    grid = np.array(cell, dtype=np.float64)
    
    # 2. Define symmetry constraints
    # The cell must be symmetric under:
    # - 90-degree rotation about center
    # - Inversion about center (180-degree rotation)
    # - Reflection across axes through center
    # For a 32x32 grid, the fundamental domain is small.
    # Let's identify the independent pixels.
    # Center is at (15.5, 15.5).
    # Symmetry group D4.
    # We will generate the full grid from a set of "master" pixels.
    
    # Helper to apply symmetries to a grid
    def apply_symmetries(master_grid):
        """
        Given a grid where only the top-left 1/8th (roughly) is defined,
        fill the rest using D4 symmetries.
        Actually, it's easier to define the full grid and ensure consistency,
        or define a fundamental domain.
        
        Let's use the fundamental domain of the square lattice unit cell under D4.
        The unit cell is [0,1)x[0,1).
        The D4 group acts on this square.
        The fundamental domain is the triangle with vertices (0,0), (0.5,0), (0.5,0.5)?
        No, the symmetry is about the cell center (0.5, 0.5).
        
        Let's just work with the full 32x32 grid but only flip pixels that are
        "independent". We can define a set of indices such that changing one
        changes its symmetry partners.
        """
        pass

    # A simpler approach for local search:
    # Pick a random pixel (i,j).
    # Flip it.
    # Flip its symmetry partners to maintain symmetry.
    # Calculate the score.
    # Accept based on Metropolis criterion.
    
    # Precompute symmetry partners for each pixel (i,j)
    # Size N=32. Center is between 15 and 16.
    # Indices 0..31.
    # Symmetries:
    # 1. Identity: (i, j)
    # 2. Rotate 90: (j, N-1-i)  (assuming (0,0) is top-left, y down, x right)
    #    Wait, standard image coords: i=row (y), j=col (x).
    #    Rotation 90 deg clockwise: new_i = j, new_j = N-1-i.
    # 3. Rotate 180: (N-1-i, N-1-j)
    # 4. Rotate 270: (N-1-j, i)
    # 5. Reflect x (vertical axis): (i, N-1-j)
    # 6. Reflect y (horizontal axis): (N-1-i, j)
    # 7. Reflect diag 1: (j, i)
    # 8. Reflect diag 2: (N-1-j, N-1-i)
    
    N = size
    partners = {}
    for i in range(N):
        for j in range(N):
            inds = set()
            inds.add((i, j))
            inds.add((j, N-1-i))
            inds.add((N-1-i, N-1-j))
            inds.add((N-1-j, i))
            inds.add((i, N-1-j))
            inds.add((N-1-i, j))
            inds.add((j, i))
            inds.add((N-1-j, N-1-i))
            # Convert to sorted tuple for consistency
            partners[(i, j)] = tuple(sorted(inds))
            
    # Get list of unique partner sets (independent pixels)
    seen_sets = set()
    independent_pixels = []
    for i in range(N):
        for j in range(N):
            pset = partners[(i, j)]
            if pset not in seen_sets:
                seen_sets.add(pset)
                independent_pixels.append(pset)
                
    # 3. Fast Surrogate PWE Solver
    # We need to estimate the gap between band 4 and 5.
    # Using a small number of plane waves (e.g., 3x3 or 5x5 grid of G vectors)
    # and a few k-points (Gamma, M, K, X).
    
    # G vectors for square lattice a=1.
    # G = (h, k) * 2*pi.
    # We'll use a cutoff |G| < G_max.
    G_max = 6.0 # Cutoff in units of 2*pi/a
    G_vectors = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            if np.sqrt(h*h + k*k) <= G_max / (2*np.pi):
                G_vectors.append((h, k))
    G_vectors = np.array(G_vectors)
    N_pw = len(G_vectors)
    
    # Precompute Fourier coefficients of epsilon for the initial grid
    # epsilon(x) = eps_low + (eps_high - eps_low) * sum_G c_G exp(i G.x)
    # For a binary pixel grid, c_G = (1/N^2) sum_{i,j} grid[i,j] exp(-i G . r_{ij})
    # r_{ij} = ((j+0.5)/N, (i+0.5)/N)
    
    def compute_fourier_coeffs(grid):
        """
        Compute Fourier coefficients c_G for the given grid.
        Returns complex array of shape (N_pw,).
        """
        coeffs = np.zeros(N_pw, dtype=np.complex128)
        for idx, (h, k) in enumerate(G_vectors):
            # G = (h, k) * 2*pi
            # Sum over pixels
            # c_G = (1/N^2) sum_{i,j} eps(i,j) exp(-i 2*pi (h*j/N + k*i/N))
            # Note: x = j/N, y = i/N.
            # exp(-i 2*pi (h*x + k*y))
            
            # Vectorized computation
            # j indices 0..N-1, i indices 0..N-1
            # We can use FFT if we pad, but N=32 is small.
            # Let's do it directly with numpy broadcasting.
            
            # Create meshgrid of j and i
            # j = np.arange(N)[:, None] -> (N, 1)
            # i = np.arange(N)[None, :] -> (1, N)
            # This is slow if done for every G.
            
            # Better: Precompute the exponential basis matrix.
            # Basis[i, j, g] = exp(-i 2*pi (h_g * j/N + k_g * i/N))
            # This is N*N*N_pw complex numbers. 32*32*~40 = 40k. Fine.
            
            pass # We'll precompute the basis outside.
            
        return coeffs

    # Precompute Basis
    # Shape: (N, N, N_pw)
    # Basis[i, j, g] = exp(-i 2*pi (h_g * (j+0.5)/N + k_g * (i+0.5)/N))
    # Note: pixel center is at (j+0.5)/N.
    
    i_idx = np.arange(N)[:, None] # (N, 1)
    j_idx = np.arange(N)[None, :] # (1, N)
    
    h_vals = G_vectors[:, 0]
    k_vals = G_vectors[:, 1]
    
    # phase = 2*pi * (h * (j+0.5)/N + k * (i+0.5)/N)
    # (j+0.5)/N is (1, N)
    # (i+0.5)/N is (N, 1)
    
    # We want shape (N, N, N_pw)
    # term_j = 2*pi * h_vals[None, None, :] * ((j_idx + 0.5)/N)[:, None, :] 
    # term_i = 2*pi * k_vals[None, None, :] * ((i_idx + 0.5)/N)[:, :, None]
    
    # Let's be careful with dimensions.
    # i_idx: (N, 1)
    # j_idx: (1, N)
    # h_vals: (N_pw,)
    # k_vals: (N_pw,)
    
    # y_part = (i_idx + 0.5) / N  # (N, 1)
    # x_part = (j_idx + 0.5) / N  # (1, N)
    
    # phase_y = 2 * np.pi * k_vals[None, None, :] * y_part[:, :, None] # (N, 1, N_pw)
    # phase_x = 2 * np.pi * h_vals[None, None, :] * x_part[None, :, :] # (1, N, N_pw)
    
    # phase = phase_y + phase_x # (N, N, N_pw)
    # basis = np.exp(-1j * phase)
    
    y_part = (i_idx + 0.5) / N
    x_part = (j_idx + 0.5) / N
    
    phase_y = 2 * np.pi * k_vals[None, None, :] * y_part[:, :, None]
    phase_x = 2 * np.pi * h_vals[None, None, :] * x_part[None, :, :]
    
    phase = phase_y + phase_x
    basis = np.exp(-1j * phase) # (N, N, N_pw)
    
    def get_coeffs(grid_flat):
        # grid_flat: (N*N,)
        # coeffs[g] = (1/N^2) sum_{i,j} grid[i,j] basis[i,j,g]
        # grid_2d = grid_flat.reshape(N, N)
        # coeffs = (1/N^2) * np.sum(grid_2d[:, :, None] * basis, axis=(0, 1))
        # Using einsum for speed?
        # coeffs = np.einsum('ij,ijg->g', grid_2d, basis) / (N*N)
        grid_2d = grid_flat.reshape(N, N)
        coeffs = np.sum(grid_2d[:, :, None] * basis, axis=(0, 1)) / (N * N)
        return coeffs

    # K-points to sample
    # Irreducible BZ: 0 <= ky <= kx, kx, ky >= 0
    # Gamma (0,0), M (0.5, 0), K (0.5, 0.5), X (0.5, 0) is M.
    # Points: Gamma, M, K, and some interior points.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.4, 0.2)
    ]
    
    def solve_bands_fast(coeffs):
        """
        Solve for the lowest 5 TM bands at the given k-points.
        Returns the min of omega_5 and max of omega_4 to calculate gap.
        """
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        # Inverse epsilon matrix
        # For TM, the eigenvalue problem is:
        # [k^2] A = [G' - G] . [1/eps(G'-G)] . [G] A
        # Where 1/eps is the Fourier transform of 1/epsilon(r).
        # This is tricky for binary media because 1/eps is not linear.
        # Standard PWE for TM uses 1/eps.
        # We need the Fourier coefficients of 1/eps(r).
        
        # Let's compute 1/eps_grid first.
        # grid is 0 or 1.
        # eps = 1 + (12-1)*grid
        # 1/eps = 1 / (1 + 11*grid)
        
        # We can compute the Fourier coeffs of 1/eps using the same basis.
        # But we don't have the grid in this function, only coeffs of eps.
        # We need to pass the grid or compute 1/eps coeffs separately.
        
        # Let's modify the interface to pass the grid.
        pass

    # Revised Fast Solver
    def evaluate_grid(grid_2d):
        """
        Evaluate the gap for a given 2D grid.
        """
        # Compute Fourier coeffs of 1/eps
        # eps(r) = eps_low + (eps_high - eps_low) * I(r)
        # 1/eps(r) = 1 / (eps_low + (eps_high - eps_low) * I(r))
        
        # For binary:
        # If I=0, 1/eps = 1/eps_low
        # If I=1, 1/eps = 1/eps_high
        
        inv_eps_grid = np.zeros_like(grid_2d, dtype=np.float64)
        inv_eps_grid[grid_2d == 0] = 1.0 / eps_low
        inv_eps_grid[grid_2d == 1] = 1.0 / eps_high
        
        # Compute Fourier coeffs of inv_eps
        # coeffs_inv[g] = (1/N^2) sum inv_eps[i,j] basis[i,j,g]
        coeffs_inv = np.sum(inv_eps_grid[:, :, None] * basis, axis=(0, 1)) / (N * N)
        
        # Build the matrix for each k-point
        # M_{mn} = (k_m - k_n) . inv_eps(k_m - k_n) . (k_m - k_n)
        # Where k_m = k + G_m
        # k_m - k_n = G_m - G_n
        
        # Precompute G differences?
        # G_diff[m, n] = G_m - G_n
        # inv_eps_diff[m, n] = coeffs_inv[G_m - G_n]
        
        # Map G vectors to indices
        # G_vectors is a list of (h,k).
        # We need to find the index of G_m - G_n in G_vectors.
        # If G_m - G_n is not in G_vectors, the coefficient is 0?
        # No, the Fourier series is truncated. If the difference is outside the truncation,
        # we assume the coefficient is 0 (which is an approximation).
        
        # Create a dictionary for fast lookup
        g_to_idx = {}
        for idx, (h, k) in enumerate(G_vectors):
            g_to_idx[(h, k)] = idx
            
        # Precompute the matrix structure?
        # The matrix depends on k.
        # M_{mn}(k) = (k + G_m - (k + G_n)) . inv_eps(G_m - G_n) . (k + G_m - (k + G_n))
        # = (G_m - G_n) . inv_eps(G_m - G_n) . (G_m - G_n)  <-- Wait, this is for k=0?
        # No, the operator is (k+G_m) x [1/eps] (k+G_n)
        # The 1/eps part acts on the difference G_m - G_n.
        # The vector dot product is (k+G_m) dot (k+G_n)? No.
        
        # The TM eigenvalue equation is:
        # sum_n [ (k+G_m) dot (k+G_n) * inv_eps(G_m - G_n) ] c_n = omega^2 c_m
        # Wait, the standard form is:
        # [ (k+G_m) dot (k+G_n) / epsilon(G_m - G_n) ] c_n = omega^2 c_m
        # Where epsilon(G) are the Fourier coeffs of epsilon, not 1/epsilon.
        # FOR TM POLARIZATION, THE DIELECTRIC TENSOR IS SCALAR, SO IT IS 1/epsilon IN THE OPERATOR.
        # Let's double check.
        # Maxwell: curl curl E = k0^2 eps E
        # In Fourier: (k x (k x E)) = k0^2 eps E
        # k x (k x E) = k(k.E) - E k^2
        # For TM (E_z = 0, E in xy plane), the operator is more complex.
        # Actually, for TM in 2D, E = (0, 0, E_z).
        # The equation is:
        # -∇^2 E_z = k0^2 eps E_z
        # In Fourier: |k+G|^2 E_z(k+G) = k0^2 sum_{G'} eps(G-G') E_z(k+G')
        # So the matrix is:
        # M_{mn} = |k+G_m|^2 delta_{mn} - k0^2 eps(G_m - G_n)
        # Eigenvalues are omega^2 / c^2 ?
        # Usually normalized:
        # M_{mn} = (k+G_m) dot (k+G_n) ... No.
        
        # Let's stick to the standard PWE formulation for TM:
        # The eigenvalue problem is:
        # [ (k+G_m) dot (k+G_n) ] * (1/eps(G_m - G_n)) c_n = (omega a / 2pi)^2 c_m
        # Wait, this is for TE?
        
        # Reference: Joannopoulos, Yeh.
        # TM:
        # sum_{G'} (k+G) dot (k+G') (1/eps(G-G')) a_{G'} = (omega a / 2pi)^2 a_G
        # TE:
        # sum_{G'} |k+G| |k+G'| (eps(G-G')) a_{G'} = (omega a / 2pi)^2 a_G
        
        # So for TM, we use 1/eps Fourier coefficients.
        # And the matrix element is:
        # M_{mn} = (k+G_m) dot (k+G_n) * inv_eps_coeffs[G_m - G_n]
        
        # Let's implement this.
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky]) * 2 * np.pi
            
            # Build Matrix M
            M = np.zeros((N_pw, N_pw), dtype=np.complex128)
            
            for m in range(N_pw):
                km = k_vec + G_vectors[m] * 2 * np.pi
                for n in range(N_pw):
                    kn = k_vec + G_vectors[n] * 2 * np.pi
                    # Dot product
                    dot = np.dot(km, kn)
                    
                    # Difference G_m - G_n
                    diff = G_vectors[m] - G_vectors[n]
                    diff_tuple = (diff[0], diff[1])
                    
                    # Get inv_eps coefficient
                    if diff_tuple in g_to_idx:
                        idx_diff = g_to_idx[diff_tuple]
                        val_inv_eps = coeffs_inv[idx_diff]
                    else:
                        val_inv_eps = 0.0
                        
                    M[m, n] = dot * val_inv_eps
                    
            # Solve eigenvalues
            # M is Hermitian?
            # inv_eps is real and symmetric (if grid is real and symmetric).
            # dot is real.
            # So M should be Hermitian.
            # Use eigh for Hermitian matrices.
            try:
                eigvals = np.linalg.eigvalsh(M)
                # eigvals are (omega a / 2pi)^2
                # We want omega a / 2pi
                # Sort them
                eigvals = np.sort(eigvals)
                # Take the first 5
                if len(eigvals) >= 5:
                    # Band 4 is index 3, Band 5 is index 4
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except np.linalg.LinAlgError:
                continue
        
        if min_omega_5 < np.inf and max_omega_4 > -np.inf:
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2
            if midgap > 0:
                return gap / midgap
            else:
                return 0.0
        else:
            return 0.0

    # 4. Optimization Loop
    # Start with the initial grid
    current_grid = grid.copy()
    current_score = evaluate_grid(current_grid)
    best_grid = current_grid.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.1
    cooling = 0.995
    
    # Get list of independent pixels (sets of indices)
    # We will iterate over these
    
    # To speed up, we can precompute the basis for each independent pixel set?
    # No, we just flip the bits in the grid and recompute coeffs.
    # Recomputing coeffs is O(N^2 * N_pw).
    # N=32, N_pw~40. 32*32*40 = 40k ops.
    # Solving eigenvalues is O(N_pw^3) = 40^3 = 64k ops.
    # Total ~100k ops per evaluation.
    # 10,000 evaluations = 10^9 ops. Might be too slow in Python.
    
    # Optimization:
    # Only update the Fourier coefficients for the changed pixels.
    # Delta_coeffs = (1/N^2) * sum_{(i,j) in flip_set} (new_val - old_val) * basis[i,j,:]
    
    def update_coeffs_delta(coeffs, flip_set, old_val, new_val):
        """
        Update coeffs by flipping pixels in flip_set from old_val to new_val.
        """
        delta = new_val - old_val
        if delta == 0:
            return coeffs
        
        # Sum basis for all pixels in flip_set
        # flip_set is a list of (i, j)
        # We need to sum basis[i, j, :] for all (i, j) in flip_set
        # And multiply by delta / (N^2)
        
        # Precomputed basis is (N, N, N_pw)
        # We can use advanced indexing
        # i_list = [p[0] for p in flip_set]
        # j_list = [p[1] for p in flip_set]
        
        # This creates a view? No, it creates a copy.
        # selected_basis = basis[i_list, j_list, :] # (len(flip_set), N_pw)
        # sum_basis = np.sum(selected_basis, axis=0)
        # coeffs += delta / (N*N) * sum_basis
        
        # For small flip sets (size 1 to 8), this is fast.
        
        i_list = np.array([p[0] for p in flip_set])
        j_list = np.array([p[1] for p in flip_set])
        
        selected_basis = basis[i_list, j_list, :]
        sum_basis = np.sum(selected_basis, axis=0)
        
        coeffs_new = coeffs + (delta / (N * N)) * sum_basis
        return coeffs_new

    # We need to maintain the current coeffs to avoid recomputing from scratch.
    current_coeffs = get_coeffs(current_grid.flatten())
    
    # Note: evaluate_grid currently computes coeffs from grid.
    # We should modify evaluate_grid to take coeffs.
    
    def evaluate_from_coeffs(coeffs):
        """
        Evaluate gap from Fourier coefficients of 1/eps.
        """
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky]) * 2 * np.pi
            
            M = np.zeros((N_pw, N_pw), dtype=np.complex128)
            
            for m in range(N_pw):
                km = k_vec + G_vectors[m] * 2 * np.pi
                for n in range(N_pw):
                    kn = k_vec + G_vectors[n] * 2 * np.pi
                    dot = np.dot(km, kn)
                    
                    diff = G_vectors[m] - G_vectors[n]
                    diff_tuple = (diff[0], diff[1])
                    
                    if diff_tuple in g_to_idx:
                        idx_diff = g_to_idx[diff_tuple]
                        val_inv_eps = coeffs[idx_diff]
                    else:
                        val_inv_eps = 0.0
                        
                    M[m, n] = dot * val_inv_eps
                    
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals)
                if len(eigvals) >= 5:
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except np.linalg.LinAlgError:
                continue
        
        if min_omega_5 < np.inf and max_omega_4 > -np.inf:
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2
            if midgap > 0:
                return gap / midgap
            else:
                return 0.0
        else:
            return 0.0

    current_score = evaluate_from_coeffs(current_coeffs)
    best_score = current_score
    best_grid = current_grid.copy()
    
    # Local Search
    # Iterate over independent pixels
    # Randomly permute them? Or just cycle through.
    
    # To avoid getting stuck in local minima, use Simulated Annealing.
    
    # Time check
    while time.time() < deadline:
        # Pick a random independent pixel set
        # To ensure we visit all, we can shuffle the list periodically.
        # For now, random choice.
        
        pset = random.choice(independent_pixels)
        
        # Determine current value of these pixels
        # They should all have the same value?
        # In a symmetric grid, all partners have the same value.
        # Let's check.
        i0, j0 = pset[0]
        current_val = current_grid[i0, j0]
        new_val = 1.0 - current_val
        
        # Apply flip
        for (i, j) in pset:
            current_grid[i, j] = new_val
            
        # Update coeffs
        new_coeffs = update_coeffs_delta(current_coeffs, pset, current_val, new_val)
        
        # Evaluate
        new_score = evaluate_from_coeffs(new_coeffs)
        
        # Metropolis
        if new_score > current_score:
            current_score = new_score
            current_coeffs = new_coeffs
            if new_score > best_score:
                best_score = new_score
                best_grid = current_grid.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Note: scores can be negative.
            # If scores are small, T needs to be small.
            # Let's use a relative difference or just a fixed T.
            # Gap ratios are typically 0 to 0.5.
            # T = 0.01 might be good.
            
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / 0.01)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current_score = new_score
                current_coeffs = new_coeffs
            else:
                # Revert
                for (i, j) in pset:
                    current_grid[i, j] = current_val
                # current_coeffs remains unchanged
                
        # Cool down
        T *= 0.999
        if T < 0.001:
            T = 0.001

    # Convert best_grid back to list of lists
    best_cell = [[int(x) for x in row] for row in best_grid]
    
    return best_cell

# EVOLVE-BLOCK-END
