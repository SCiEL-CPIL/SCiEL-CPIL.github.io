import numpy as np
import time
import sys

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Define Symmetry Orbits for the 32x32 grid
    # We only need to define the top-left 8x8 quadrant (indices 0..7, 0..7)
    # because of C4v symmetry (rotations and reflections).
    # The quadrant is 8x8. Total unique pixels = 64.
    # However, the symmetry maps (i, j) in 0..31 to other positions.
    # Let's define the orbits properly for the whole grid to ensure consistency.
    
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            # C4v group elements:
            # (i, j)
            # (j, size-1-i)  # 90 deg rot
            # (size-1-i, size-1-j) # 180 deg rot
            # (size-1-j, i)   # 270 deg rot
            # (i, size-1-j)   # Reflect X (vertical axis)
            # (size-1-i, j)   # Reflect Y (horizontal axis)
            # (j, i)          # Reflect Diagonal
            # (size-1-j, size-1-i) # Reflect Anti-Diagonal
            
            points = [
                (i, j),
                (j, size-1-i),
                (size-1-i, size-1-j),
                (size-1-j, i),
                (i, size-1-j),
                (size-1-i, j),
                (j, i),
                (size-1-j, size-1-i)
            ]
            for p in points:
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

    # 2. Fast Plane Wave Evaluator
    # Use a truncated basis. 17x17 = 289 modes is the evaluator's size.
    # We can use 17x17 for high accuracy, or 11x11 (121 modes) for speed.
    # Given 120s budget, 17x17 eigh is ~5-10ms per k-point.
    # 30 k-points * 10ms = 0.3s per evaluation.
    # This allows ~300 evaluations in 100s.
    
    N_pw = 17
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y], dtype=np.float64)
    N_modes = len(G_vectors)
    
    # Precompute indices for epsilon matrix
    # eps_mat[m, n] = eps_G (G_m - G_n)
    # G_diff = G_m - G_n
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    # Map G_diff to pixel indices (0..31)
    # Note: G vectors are integers. Pixel indices are 0..31.
    # The Fourier coefficient epsilon_G corresponds to the sum over pixels of eps_r * exp(-i G . r)
    # In the plane wave method, the matrix element involves epsilon_{G_m - G_n}.
    # We compute the FFT of the epsilon grid to get epsilon_G.
    
    # K-points in the irreducible Brillouin zone (square lattice)
    # The irreducible BZ is 1/8 of the full BZ.
    # We sample points to cover the boundaries Gamma-X, Gamma-M, X-M.
    # Gamma = (0,0), X = (0.5, 0), M = (0.5, 0.5)
    # Path: Gamma -> X -> M -> Gamma
    k_points = []
    # Along Gamma-X (y=0, x: 0->0.5)
    for i in range(6):
        k_points.append((i * 0.5 / 5, 0.0))
    # Along X-M (x=0.5, y: 0->0.5)
    for i in range(1, 6):
        k_points.append((0.5, i * 0.5 / 5))
    # Along M-Gamma (x: 0.5->0, y: 0.5->0)
    for i in range(1, 6):
        k_points.append((0.5 - i * 0.5 / 5, 0.5 - i * 0.5 / 5))
    # Add a few interior points for better gap estimation
    k_points.append((0.25, 0.25))
    k_points.append((0.1, 0.1))
    k_points.append((0.4, 0.1))
    k_points.append((0.1, 0.4))
    
    def evaluate(orbit_vals):
        """
        Returns the gap ratio.
        """
        grid = orbit_to_grid(orbit_vals)
        eps_grid = eps_low + (eps_high - eps_low) * grid
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0.0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            kG = G_vectors + np.array([kx, ky])
            kG_sq = np.sum(kG**2, axis=1)
            dot = np.dot(kG, kG.T)
            
            # Construct epsilon matrix
            # eps_mat[m, n] = eps_fft[G_m - G_n]
            # G_diff is in integer units. We need to map to 0..31 indices.
            # Since size=32, we can take modulo 32.
            idx_x = G_diff[:, :, 0] % size
            idx_y = G_diff[:, :, 1] % size
            eps_mat = eps_fft[idx_x, idx_y].real
            
            # M = diag(kG_sq) - dot / eps_mat
            # Note: dot / eps_mat is element-wise division
            M = np.diag(kG_sq) - dot / eps_mat
            
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
            except:
                return -1.0
                
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
        return gap / midgap

    # 3. Optimization Strategy
    
    # Phase 1: Coarse Parametric Search
    # Define a small set of geometric primitives.
    # 1. Central square rod (size s1)
    # 2. Outer square ring (inner size s2, outer size s3)
    # 3. Cross bars (width w)
    
    def generate_params(params):
        """
        params: [s1, s2, s3, w]
        s1: half-width of central square (0..7)
        s2: inner half-width of ring (0..7)
        s3: outer half-width of ring (0..7)
        w: width of cross bars (0..7)
        """
        s1, s2, s3, w = params
        grid = np.zeros((size, size), dtype=np.float64)
        
        # Central square
        for i in range(size):
            for j in range(size):
                # Center is (15.5, 15.5)
                # Distance to nearest edge of central square
                dx = abs(i - 15.5)
                dy = abs(j - 15.5)
                if dx <= s1 and dy <= s1:
                    grid[i, j] = 1.0
        
        # Outer ring
        for i in range(size):
            for j in range(size):
                dx = abs(i - 15.5)
                dy = abs(j - 15.5)
                # Inside outer boundary
                if dx <= s3 and dy <= s3:
                    # Outside inner boundary
                    if not (dx <= s2 and dy <= s2):
                        grid[i, j] = 1.0
        
        # Cross bars
        # Vertical bar: |x| <= w/2
        # Horizontal bar: |y| <= w/2
        for i in range(size):
            for j in range(size):
                dx = abs(i - 15.5)
                dy = abs(j - 15.5)
                if dx <= w or dy <= w:
                    # Only add if not already set? Or force?
                    # Forcing might break the ring structure.
                    # Let's just set to 1.
                    grid[i, j] = 1.0
        
        # Convert to orbit values
        vals = np.zeros(n_orbits)
        for k in range(n_orbits):
            s = 0
            cnt = 0
            for (i, j) in orbits[k]:
                s += grid[i, j]
                cnt += 1
            vals[k] = s / cnt
        return vals

    # Grid search over the 4 parameters
    # s1: 0..7
    # s2: 0..7
    # s3: 0..7
    # w: 0..7
    # Total 8^4 = 4096 combinations.
    # Evaluation cost ~0.3s. 4096 * 0.3 = 1228s. Too slow.
    # We need to reduce the search space or use a smarter method.
    
    # Let's use a random search with a time limit.
    start_time = time.time()
    deadline = start_time + 90.0 # Leave 30s for refinement
    
    best_score = -np.inf
    best_vals = None
    
    # Initialize with a few known good structures
    initial_candidates = [
        [3, 5, 7, 1], # Rod + Ring + Thin Cross
        [4, 6, 7, 2], # Larger Rod + Ring + Thicker Cross
        [2, 4, 6, 3], # Small Rod + Ring + Thick Cross
        [5, 7, 7, 0], # Large Rod + Ring, No Cross
        [0, 5, 7, 4], # No Rod, Ring, Thick Cross
    ]
    
    for params in initial_candidates:
        vals = generate_params(params)
        score = evaluate(vals)
        if score > best_score:
            best_score = score
            best_vals = vals.copy()
            
    # Random search
    np.random.seed(42)
    while time.time() < deadline:
        # Random parameters
        s1 = np.random.randint(0, 8)
        s2 = np.random.randint(0, 8)
        s3 = np.random.randint(s2, 8) # s3 >= s2
        w = np.random.randint(0, 8)
        
        params = [s1, s2, s3, w]
        vals = generate_params(params)
        score = evaluate(vals)
        
        if score > best_score:
            best_score = score
            best_vals = vals.copy()
            
        # Occasionally try a perturbation of the best
        if np.random.rand() < 0.3 and best_vals is not None:
            perturbed = best_vals.copy()
            # Flip a random orbit
            idx = np.random.randint(n_orbits)
            perturbed[idx] = 1 - perturbed[idx]
            score = evaluate(perturbed)
            if score > best_score:
                best_score = score
                best_vals = perturbed.copy()

    # Phase 2: Local Refinement (Pixel Flipping)
    # Take the best parametric design and refine it by flipping individual orbits.
    # This allows the structure to deviate from the strict parametric family.
    
    current_vals = best_vals.copy()
    current_score = best_score
    
    # Hill Climbing
    while time.time() < deadline + 25.0: # Use remaining time
        improved = False
        # Shuffle orbits for diversity
        orbit_indices = np.arange(n_orbits)
        np.random.shuffle(orbit_indices)
        
        for idx in orbit_indices:
            if time.time() > deadline + 25.0:
                break
                
            new_vals = current_vals.copy()
            new_vals[idx] = 1 - new_vals[idx]
            new_score = evaluate(new_vals)
            
            if new_score > current_score:
                current_vals = new_vals
                current_score = new_score
                improved = True
                
        if not improved:
            # Stuck in local minimum. Try a few random restarts from best?
            # Or just break.
            # Let's try a few random perturbations of the best found so far.
            for _ in range(5):
                if time.time() > deadline + 25.0:
                    break
                perturbed = current_vals.copy()
                for _ in range(10):
                    idx = np.random.randint(n_orbits)
                    perturbed[idx] = 1 - perturbed[idx]
                score = evaluate(perturbed)
                if score > current_score:
                    current_vals = perturbed
                    current_score = score
            break

    # Convert best_vals to grid
    best_grid = orbit_to_grid(current_vals)
    
    # Threshold to 0/1
    # Since we started with 0/1 and flipped, values should be 0 or 1.
    # But orbit_to_grid averages if there's inconsistency? No, we maintain consistency.
    # Just to be safe:
    best_grid = (best_grid > 0.5).astype(int)
    
    return best_grid.tolist()
