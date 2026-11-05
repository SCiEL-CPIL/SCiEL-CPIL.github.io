import numpy as np
import time
import sys

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Pre-compute symmetry orbits
    # Center is at (size/2 - 0.5, size/2 - 0.5)
    # Symmetries of the square cell centered at origin (shifted):
    # (x, y) -> (y, size-1-x) [90 deg rot]
    # (x, y) -> (x, size-1-y) [refl y]
    
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            for _ in range(4):
                orbit.add((x, y))
                orbit.add((x, size - 1 - y))
                x, y = y, size - 1 - x
            orbits.append(orbit)
            seen.update(orbit)
            
    # Function to apply a set of orbit values to a grid
    def orbit_to_grid(orbit_vals):
        grid = np.zeros((size, size), dtype=int)
        for k, orbit in enumerate(orbits):
            val = orbit_vals[k]
            for (i, j) in orbit:
                grid[i, j] = val
        return grid

    # Initial design: Rod
    def make_initial():
        grid = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        radius = 0.2 * size # 0.2 lattice constants, but here lattice constant is size pixels? 
        # Prompt says "radius 0.2 lattice constants". 
        # In the seed: x = (i+0.5)/size - 0.5. This maps pixel center to [-0.5, 0.5].
        # So radius 0.2 in this coordinate system.
        # Distance from center (0,0) in this system.
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    grid[i, j] = 1
        # Convert grid to orbit values
        vals = []
        for orbit in orbits:
            # Check if all pixels in orbit are same (they should be for a symmetric seed)
            # If not, take majority or first
            first_val = grid[list(orbit)[0][0], list(orbit)[0][1]]
            vals.append(first_val)
        return np.array(vals, dtype=int)

    current_vals = make_initial()
    best_vals = current_vals.copy()
    
    # Fast Evaluator
    # We will use a reduced plane wave set for speed.
    # N_pw = 7 (from -3 to 3) -> 49 modes.
    # K-points: Gamma, M, X, and midpoints.
    
    # Pre-compute G vectors
    N_pw = 7
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in the irreducible Brillouin zone (1/8 of the BZ)
    # BZ is [-0.5, 0.5] x [-0.5, 0.5] in units of 2pi/a.
    # Irreducible wedge: 0 <= ky <= kx, kx <= 0.5.
    # Points:
    # Gamma: (0, 0)
    # M: (0.5, 0.5)
    # X: (0.5, 0)
    # Mid-Gamma-M: (0.25, 0.25)
    # Mid-Gamma-X: (0.25, 0)
    # Mid-X-M: (0.5, 0.25)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25)
    ]
    
    def fast_evaluate(vals):
        """
        Calculate the gap metric for a given orbit value vector.
        """
        # Reconstruct grid
        grid = orbit_to_grid(vals)
        
        # Compute Fourier coefficients epsilon_G
        # epsilon(r) = eps_low + (eps_high - eps_low) * grid(r)
        eps_grid = eps_low + (eps_high - eps_low) * grid
        
        # FFT to get coefficients
        # The coefficient for G=(gx, gy) is 1/N^2 sum eps(r) exp(-i G . r)
        # Using numpy fft2, the index [i, j] corresponds to frequency (i, j) if i,j < N/2, else (i-N, j-N).
        # Our G vectors are in range [-3, 3].
        # We need to map our G_vectors to FFT indices.
        # FFT index for G=(gx, gy) is (gx % size, gy % size).
        
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        # Pre-compute matrix parts that don't depend on k
        # We need epsilon_{G-G'}
        # For each pair of G, G', we need eps_fft[ (Gx-G'x)%size, (Gy-G'y)%size ]
        
        min_gap_ratio = float('inf')
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        # We want to find the max omega_4 and min omega_5 across the BZ
        # Actually, the gap is defined as min(omega_5) - max(omega_4) over the BZ.
        # So we need to track the global max of the 4th band and global min of the 5th band.
        
        global_max_4 = 0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            # Build the PWE matrix
            # A_{G, G'} = delta_{G, G'} * (k+G)^2 / c^2 - 1/c^2 * (k+G).(k+G') / epsilon_{G-G'}
            # Wait, the standard TM PWE equation is:
            # sum_{G'} [ (k+G)^2 delta_{G,G'} - (k+G).(k+G') / epsilon_{G-G'} ] c_{G'} = omega^2 c_G
            # This is a generalized eigenvalue problem if we don't divide by epsilon.
            # Or we can define the matrix M such that M c = omega^2 c.
            # M_{G, G'} = (k+G)^2 delta_{G,G'} - (k+G).(k+G') / epsilon_{G-G'}
            # This requires epsilon_{G-G'} to be non-zero. Since eps >= 1, it's fine.
            
            M = np.zeros((N_modes, N_modes), dtype=float)
            
            for i in range(N_modes):
                gx, gy = G_vectors[i]
                kx_i = kx + gx
                ky_i = ky + gy
                k_sq_i = kx_i**2 + ky_i**2
                
                for j in range(N_modes):
                    gx_j, gy_j = G_vectors[j]
                    kx_j = kx + gx_j
                    ky_j = ky + gy_j
                    
                    # Dot product (k+G_i) . (k+G_j)
                    dot = kx_i * kx_j + ky_i * ky_j
                    
                    # Epsilon coefficient
                    # G_diff = G_i - G_j
                    dgx = gx - gx_j
                    dgy = gy - gy_j
                    # Map to FFT index
                    idx_i = dgx % size
                    idx_j = dgy % size
                    eps_val = eps_fft[idx_i, idx_j].real # Should be real for symmetric grid?
                    # Note: For a real grid, eps_{-G} = eps_G^*. The matrix should be symmetric/Hermitian.
                    # Using .real is a good approximation for symmetric structures, but let's be careful.
                    # Actually, the matrix M is symmetric real if the structure is symmetric.
                    
                    if i == j:
                        M[i, j] = k_sq_i - dot / eps_val
                    else:
                        M[i, j] = - dot / eps_val
                        
            # Solve for eigenvalues
            # We only need the 4th and 5th lowest eigenvalues.
            # numpy.linalg.eigvalsh is efficient for symmetric matrices.
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0 # Error
                
            # eigvals are sorted ascending.
            # We need the 4th and 5th bands. Indices 3 and 4 (0-indexed).
            # Note: The PWE method usually gives positive frequencies.
            # The lowest modes are the physical ones.
            
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
            if omega_4 > global_max_4:
                global_max_4 = omega_4
            if omega_5 < global_min_5:
                global_min_5 = omega_5
                
        if global_min_5 == float('inf'):
            return -1.0
            
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    # Optimization Loop
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    current_score = fast_evaluate(current_vals)
    best_score = current_score
    best_vals = current_vals.copy()
    
    # Simulated Annealing parameters
    T = 0.05
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random orbit to flip
        idx = np.random.randint(len(orbits))
        
        # Flip the bit
        new_vals = current_vals.copy()
        new_vals[idx] = 1 - new_vals[idx]
        
        new_score = fast_evaluate(new_vals)
        
        if new_score > current_score:
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = current_vals.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_vals = new_vals
                    current_score = new_score
            
        T *= cooling_rate
        if T < 0.001:
            T = 0.001 # Floor temperature
            
    # Return the best grid found
    best_grid = orbit_to_grid(best_vals)
    # Convert to list of lists of ints
    return best_grid.tolist()
