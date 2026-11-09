import numpy as np
import time

def design(problem):
    """
    Optimizes a binary 32x32 dielectric pixel map using Simulated Annealing.
    The design is constrained to the C4v symmetry group (1/8 of the pixels are independent).
    The objective function approximates the TM band gap between bands 4 and 5
    using a plane-wave expansion.
    """
    start_time = time.time()
    # Safety margin: 230s kill, aim for 220s limit to ensure return
    deadline = start_time + 220.0
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Define the independent pixels (C4v symmetry)
    # The symmetry group acts on (i, j). We pick the canonical representative
    # for each orbit.
    # Symmetries:
    # 1. Identity: (i, j)
    # 2. Rotate 90: (j, size-1-i)
    # 3. Rotate 180: (size-1-i, size-1-j)
    # 4. Rotate 270: (size-1-j, i)
    # 5. Reflect x (i -> size-1-i): (size-1-i, j)
    # 6. Reflect y (j -> size-1-j): (i, size-1-j)
    # 7. Reflect diag (i,j -> j,i): (j, i)
    # 8. Reflect anti-diag: (size-1-j, size-1-i)
    
    independent_pixels = []
    is_independent = np.zeros((size, size), dtype=bool)
    
    for i in range(size):
        for j in range(size):
            if not is_independent[i, j]:
                # Generate all symmetric counterparts
                orbit = set()
                orbit.add((i, j))
                orbit.add((j, size - 1 - i))
                orbit.add((size - 1 - i, size - 1 - j))
                orbit.add((size - 1 - j, i))
                orbit.add((size - 1 - i, j))
                orbit.add((i, size - 1 - j))
                orbit.add((j, i))
                orbit.add((size - 1 - j, size - 1 - i))
                
                # Mark all as dependent on the first one found
                for pi, pj in orbit:
                    is_independent[pi, pj] = True
                
                # Store the representative (i, j) as the independent variable
                independent_pixels.append((i, j))
    
    num_independent = len(independent_pixels)
    
    # 2. Precomputation for Plane-Wave Evaluation
    # Use N_pw=11 (121 PWs) for a good balance of accuracy and speed.
    # 289 PWs is ~8x slower. 121 should capture the main gap features.
    N_pw = 11
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: Sample the irreducible BZ.
    # The current code uses 15 points. We will use 15 points as well.
    k_points = []
    # Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
    # M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((0.5, t))
    # Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, t))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
            
    # Precompute (k+G)^2 for each k point
    # k_sq[k_idx, pw_idx] = |k + G_pw|^2
    k_sq = np.zeros((len(unique_k), num_pw))
    for ki, (kx, ky) in enumerate(unique_k):
        k = np.array([kx, ky])
        k_plus_G = k + G_vectors
        k_sq[ki] = np.sum(k_plus_G ** 2, axis=1)
        
    # Precompute G-G' differences for coefficient lookup
    # We need to map (G_i - G_j) to FFT indices
    # coeff[i, j] = FFT[inv_eps][(G_i - G_j) mod size]
    
    def get_eps_grid(binary_independent):
        """
        Expands the independent binary pixels to the full 32x32 grid.
        binary_independent: 1D array of 0/1 of length num_independent
        """
        eps_grid = np.zeros((size, size), dtype=float)
        for idx, (i, j) in enumerate(independent_pixels):
            val = binary_independent[idx]
            # Apply symmetry
            # (i, j)
            eps_grid[i, j] = val
            # (j, size-1-i)
            eps_grid[j, size - 1 - i] = val
            # (size-1-i, size-1-j)
            eps_grid[size - 1 - i, size - 1 - j] = val
            # (size-1-j, i)
            eps_grid[size - 1 - j, i] = val
            # (size-1-i, j)
            eps_grid[size - 1 - i, j] = val
            # (i, size-1-j)
            eps_grid[i, size - 1 - j] = val
            # (j, i)
            eps_grid[j, i] = val
            # (size-1-j, size-1-i)
            eps_grid[size - 1 - j, size - 1 - i] = val
            
        # Map 0/1 to eps_low/eps_high
        eps_grid = eps_low + (eps_high - eps_low) * eps_grid
        return eps_grid

    def evaluate_score(binary_independent):
        """
        Computes the gap-midgap ratio.
        """
        eps_grid = get_eps_grid(binary_independent)
        inv_eps = 1.0 / eps_grid
        
        # FFT of 1/eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # Build the coefficient matrix C
        # C[i, j] = fft_inv_eps[(G_i - G_j) mod size]
        # We can compute this efficiently by precomputing the G differences
        # But since N_pw is small (121), a loop is okay, or we can vectorize.
        
        # Let's precompute the indices for FFT lookup
        # G_diff[i, j, 0] = Gx[i] - Gx[j]
        # G_diff[i, j, 1] = Gy[i] - Gy[j]
        
        # To speed up, we precompute the modulo indices for all pairs
        # idx_x[i, j] = (Gx[i] - Gx[j]) % size
        # idx_y[i, j] = (Gy[i] - Gy[j]) % size
        
        # This is done once outside the loop in the setup below.
        
        # For now, inside the function:
        C = np.zeros((num_pw, num_pw), dtype=complex)
        # Vectorized lookup
        # Gx, Gy are 1D arrays of length num_pw
        # We need C[i,j] = fft_inv_eps[ (Gx[i]-Gx[j])%size, (Gy[i]-Gy[j])%size ]
        
        # Create meshgrid of G indices
        Gx_col = G_vectors[:, 0][:, np.newaxis]
        Gy_col = G_vectors[:, 1][:, np.newaxis]
        
        # Differences
        # dx[i, j] = Gx[i] - Gx[j]
        # dy[i, j] = Gy[i] - Gy[j]
        
        # This creates large arrays (121x121). It's fine.
        # However, doing this in every call is wasteful.
        # We will precompute the index arrays.
        
        pass # Placeholder, logic moved to setup

    # Precompute FFT lookup indices
    # G_vectors is (num_pw, 2)
    # We want indices into fft_inv_eps (size x size)
    # idx_x[i, j] = (G_vectors[i, 0] - G_vectors[j, 0]) % size
    # idx_y[i, j] = (G_vectors[i, 1] - G_vectors[j, 1]) % size
    
    Gx_arr = G_vectors[:, 0]
    Gy_arr = G_vectors[:, 1]
    
    # Create index grids
    # This is a bit memory intensive (121x121x2 ints), but fast.
    idx_x_grid = (Gx_arr[:, np.newaxis] - Gx_arr[np.newaxis, :]) % size
    idx_y_grid = (Gy_arr[:, np.newaxis] - Gy_arr[np.newaxis, :]) % size
    
    def evaluate_score_fast(binary_independent):
        """
        Fast evaluation using precomputed indices.
        """
        eps_grid = get_eps_grid(binary_independent)
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # C[i, j] = fft_inv_eps[idx_y_grid[i, j], idx_x_grid[i, j]]
        # Note: fft2 output is indexed [row, col] which corresponds to [y, x] in standard math, 
        # but numpy fft2 is just 2D array.
        # Our G_vectors are (kx, ky).
        # fft2(inv_eps) gives F[ny, nx] = sum exp(-i 2pi (nx x + ny y) / N)
        # The coefficient for G=(Gx, Gy) and G'=(Gx', Gy') involves (G-G')
        # The term in the Hamiltonian matrix is (k+G)^2 * (1/eps)_{G-G'}
        # (1/eps)_{G-G'} is the Fourier coefficient of 1/eps at frequency G-G'.
        # In discrete FFT: coeff[m, n] = 1/N^2 sum eps_inv[x,y] exp(-i 2pi (m x + n y)/N)
        # Here m corresponds to x-frequency (Gx), n to y-frequency (Gy).
        # So we look up fft_inv_eps[ n, m ] i.e. [Gy_diff, Gx_diff].
        
        C = fft_inv_eps[idx_y_grid, idx_x_grid]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for ki in range(len(unique_k)):
            # M[i, j] = C[i, j] * k_sq[ki, i]
            # The eigenvalues of M give omega^2 (in normalized units where a=1, c=1)
            # Actually, the eigenvalue equation is:
            # sum_j (k+G_i)^2 C_ij A_j = omega^2 A_i
            # So we solve for eigenvalues of M = C * diag(k_sq) ?
            # Wait, the standard PW method for TM:
            # (k+G)^2 sum_{G'} (1/eps)_{G-G'} A_{G'} = omega^2 A_G
            # So M_{G, G'} = (k+G)^2 (1/eps)_{G-G'}
            # Yes, M[i, j] = k_sq[ki, i] * C[i, j]
            
            M = C * k_sq[ki, :][:, np.newaxis]
            
            # Solve eigenvalues
            # M is not Hermitian if 1/eps is real but C is complex?
            # 1/eps is real. FFT of real is Hermitian: C[j, i] = conj(C[i, j])
            # So M is not Hermitian because of the (k+G)^2 factor on the left.
            # However, the eigenvalues should be real.
            # Using eigvals is fine.
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # Ensure positive
                real_eigvals = np.maximum(real_eigvals, 0)
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

    # 3. Simulated Annealing Optimization
    
    # Initial guess: Random or specific pattern
    # A checkerboard or rod pattern is a good start.
    # Let's try a "rod" in the center.
    # Center is (15.5, 15.5).
    # Radius ~ 4 pixels.
    
    current_binary = np.zeros(num_independent, dtype=int)
    for idx, (i, j) in enumerate(independent_pixels):
        # Distance from center
        dx = i - 15.5
        dy = j - 15.5
        if dx*dx + dy*dy < 20: # ~radius 4.5
            current_binary[idx] = 1
            
    current_score = evaluate_score_fast(current_binary)
    best_binary = current_binary.copy()
    best_score = current_score
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    T = T_start
    
    # Number of steps
    # We have ~220s.
    # Each eval takes ~0.1-0.2s?
    # Let's measure one eval time.
    t0 = time.time()
    _ = evaluate_score_fast(current_binary)
    t_eval = time.time() - t0
    
    # Estimate steps possible
    # Leave 10s for overhead
    available_time = deadline - start_time - 10
    max_steps = int(available_time / t_eval)
    
    # If max_steps is too high, limit it to avoid excessive loop overhead
    max_steps = min(max_steps, 2000)
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
            
        # Perturb: Flip 1-3 random independent pixels
        num_flips = np.random.randint(1, 4)
        indices_to_flip = np.random.choice(num_independent, size=num_flips, replace=False)
        
        new_binary = current_binary.copy()
        new_binary[indices_to_flip] = 1 - new_binary[indices_to_flip]
        
        new_score = evaluate_score_fast(new_binary)
        
        # Acceptance criterion
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
            current_binary = new_binary
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_binary = current_binary.copy()
                
        T *= cooling_rate
        
    # 4. Final Binarization and Symmetry Enforcement
    # The SA process already maintains symmetry by construction (operating on independent pixels).
    # We just need to return the full grid.
    
    final_grid = get_eps_grid(best_binary)
    
    # Convert to 0/1 integers
    # Threshold at 0.5 * (eps_low + eps_high)
    threshold = 0.5 * (eps_low + eps_high)
    binary_cell = (final_grid > threshold).astype(int)
    
    # Double check symmetry just in case
    # The construction guarantees it, but let's be safe.
    # cell[i][j] == cell[j][i] etc.
    
    # Return as nested list
    return binary_cell.tolist()

