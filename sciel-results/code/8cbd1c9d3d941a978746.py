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
    
    # Symmetry handling
    # The cell is 32x32. Center is effectively at (15.5, 15.5) if we consider pixel indices 0..31.
    # Symmetries:
    # 1. Identity: (i, j)
    # 2. Rot 90: (j, size-1-i)
    # 3. Rot 180: (size-1-i, size-1-j)
    # 4. Rot 270: (size-1-j, i)
    # 5. Refl X (vertical axis): (i, size-1-j)
    # 6. Refl Y (horizontal axis): (size-1-i, j)
    # 7. Refl Diag: (j, i)
    # 8. Refl AntiDiag: (size-1-j, size-1-i)
    
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            # Generate orbit under the symmetry group
            points = []
            points.append((x, y))
            points.append((y, size-1-x))
            points.append((size-1-x, size-1-y))
            points.append((size-1-y, x))
            points.append((x, size-1-y))
            points.append((size-1-x, y))
            points.append((y, x))
            points.append((size-1-y, size-1-x))
            
            for p in points:
                if p not in orbit:
                    orbit.add(p)
            
            orbits.append(orbit)
            seen.update(orbit)
    
    n_orbits = len(orbits)
    
    def orbit_to_grid(orbit_vals):
        grid = np.zeros((size, size), dtype=np.float64)
        for k in range(n_orbits):
            val = orbit_vals[k]
            for (i, j) in orbits[k]:
                grid[i, j] = val
        return grid

    # Fast Evaluator
    # Use a reduced plane wave basis for speed during optimization.
    # N_pw = 15 (from -7 to 7) -> 225 modes.
    N_pw = 15
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in the irreducible Brillouin zone
    # We sample a denser set of points to approximate the gap more accurately.
    # Irreducible zone is 0 <= kx <= ky <= 0.5
    # We use a grid of points.
    k_points = []
    # Boundary points
    k_points.append((0.0, 0.0)) # Gamma
    k_points.append((0.5, 0.5)) # M
    k_points.append((0.5, 0.0)) # X
    # Interior points
    for i in range(1, 5):
        for j in range(i, 5):
            kx = i * 0.1
            ky = j * 0.1
            if kx <= ky <= 0.5:
                k_points.append((kx, ky))
    # Add some specific midpoints
    k_points.append((0.25, 0.25))
    k_points.append((0.25, 0.5))
    k_points.append((0.5, 0.25))
    # Remove duplicates
    k_points = list(set([(round(kx, 3), round(ky, 3)) for kx, ky in k_points]))

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
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            # Build the PWE matrix
            # M_{G, G'} = (k+G)^2 delta_{G,G'} - (k+G).(k+G') / epsilon_{G-G'}
            
            # Precompute (k+G) vectors and their squares
            kG = G_vectors + np.array([kx, ky])
            kG_sq = np.sum(kG**2, axis=1)
            
            # Dot product matrix
            dot = np.dot(kG, kG.T)
            
            # Epsilon matrix
            G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
            
            # Efficient way to get eps_mat
            idx_x = G_diff[:, :, 0] % size
            idx_y = G_diff[:, :, 1] % size
            eps_mat = eps_fft[idx_x, idx_y].real
            
            # M = diag(kG_sq) - dot / eps_mat
            M = np.diag(kG_sq) - dot / eps_mat
            
            # Solve for eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0
                
            if len(eigvals) < 5:
                return -1.0
                
            # Take the 4th and 5th bands (indices 3 and 4)
            # Ensure we are taking positive eigenvalues
            if eigvals[3] < 0 or eigvals[4] < 0:
                # If negative, something is wrong, but typically low bands are positive
                # For robustness, skip if invalid
                continue
                
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

    # Initial design: Hole in dielectric
    # This is often better for lower band gaps with high epsilon contrast
    def make_initial():
        grid = np.ones((size, size), dtype=int) # Start with all dielectric
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Square hole
                if abs(x) < 0.15 and abs(y) < 0.15:
                    grid[i, j] = 0
        vals = []
        for orbit in orbits:
            first_val = grid[list(orbit)[0][0], list(orbit)[0][1]]
            vals.append(first_val)
        return np.array(vals, dtype=int)

    current_vals = make_initial()
    best_vals = current_vals.copy()
    
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for safety
    
    current_score = fast_evaluate(current_vals)
    best_score = current_score
    best_vals = current_vals.copy()
    
    # Simulated Annealing parameters
    T = 0.05
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random orbit to flip
        idx = np.random.randint(n_orbits)
        
        # Flip the bit
        new_vals = current_vals.copy()
        new_vals[idx] = 1 - new_vals[idx]
        
        new_score = fast_evaluate(new_vals)
        
        if new_score > current_score:
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = new_vals.copy()
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
    return best_grid.astype(int).tolist()
