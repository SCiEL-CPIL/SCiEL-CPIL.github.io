import numpy as np
import time
import sys
from itertools import product

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
            # We can just apply the 8 transformations
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
    # We use a reduced plane wave basis for speed during optimization.
    # N_pw = 9 (from -4 to 4) -> 81 modes. This should be a good balance.
    N_pw = 9
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in the irreducible Brillouin zone
    # We sample a few key points to approximate the gap.
    # Gamma (0,0), M (0.5,0.5), X (0.5,0)
    # And midpoints to catch band crossings.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.375),
        (0.375, 0.375)
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
            # dot[i, j] = kG[i] . kG[j]
            dot = np.dot(kG, kG.T)
            
            # Epsilon matrix
            # eps_mat[i, j] = eps_{G_i - G_j}
            # G_diff[i, j] = G_i - G_j
            G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
            # Flatten to get indices
            # We need to map G_diff to FFT indices
            # FFT index is (gx % size, gy % size)
            
            # Efficient way to get eps_mat:
            # We can loop or use advanced indexing.
            # Let's try advanced indexing.
            idx_x = G_diff[:, :, 0] % size
            idx_y = G_diff[:, :, 1] % size
            eps_mat = eps_fft[idx_x, idx_y].real
            
            # M = diag(kG_sq) - dot / eps_mat
            # Note: division is element-wise
            # Handle potential division by zero? eps_low is 1, so min eps is 1.
            # So eps_mat >= 1. No zero division.
            
            M = np.diag(kG_sq) - dot / eps_mat
            
            # Solve for eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0
                
            # We need the 4th and 5th bands.
            # The PWE method might yield negative eigenvalues for non-physical modes or due to numerical issues,
            # but usually the lowest positive ones are the physical bands.
            # However, for a gap between band 4 and 5, we assume indices 3 and 4 (0-indexed) in the sorted list.
            # We should filter out negative eigenvalues if any, but typically the lowest modes are positive.
            
            # Let's take the sorted positive eigenvalues.
            # Actually, for TM modes, omega^2 > 0.
            # Let's just take eigvals[3] and eigvals[4] as they are sorted ascending.
            
            if len(eigvals) < 5:
                return -1.0
                
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

    # Initial design: Rod
    def make_initial():
        grid = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    grid[i, j] = 1
        vals = []
        for orbit in orbits:
            first_val = grid[list(orbit)[0][0], list(orbit)[0][1]]
            vals.append(first_val)
        return np.array(vals, dtype=int)

    current_vals = make_initial()
    best_vals = current_vals.copy()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 20s margin for safety
    
    current_score = fast_evaluate(current_vals)
    best_score = current_score
    best_vals = current_vals.copy()
    
    # Simulated Annealing parameters
    T = 0.02
    cooling_rate = 0.9995
    
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
