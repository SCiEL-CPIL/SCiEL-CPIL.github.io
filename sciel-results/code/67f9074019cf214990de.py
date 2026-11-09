import numpy as np
import time

def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM band 4-5 gap."""
    start_time = time.time()
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # 289 plane waves => (2*N+1)^2 = 289 => N=8
    N_pw = 8
    G_range = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # shape (289, 2)
    num_pw = len(G_vectors)
    
    # Precompute G_diff indices for Hamiltonian construction
    # H[a, b] = inv_eps_G[G_a - G_b] * |k+G_b|^2
    # We need a lookup for inv_eps_G[G_a - G_b]
    # Since G_vectors are integers in [-8, 8], G_a - G_b is in [-16, 16]
    # But our DFT grid only covers [-8, 8]. 
    # Wait, the problem says "exact Fourier coefficients of the pixel map".
    # The pixel map is 32x32. The DFT gives coefficients for G in [-15, 15] effectively (Nyquist).
    # However, the PWE method usually uses a truncated basis.
    # The Hamiltonian matrix element H_{ab} = (1/eps)_{G_a - G_b} * |k+G_b|^2.
    # If G_a - G_b is outside the range of computed Fourier coefficients, we treat it as 0?
    # Or do we need to compute more?
    # The evaluator uses 289 plane waves. This implies the basis is 17x17.
    # The Fourier coefficients of the dielectric map are periodic.
    # For a 32x32 pixel cell, the fundamental frequencies are 1/32.
    # The G vectors in the PWE basis are integers (reciprocal lattice vectors).
    # The Fourier coefficient of the pixel map at integer G is the DFT of the pixel map at G.
    # Since the pixel map is 32x32, the DFT is defined for G = -16..16 (or 0..31).
    # The PWE basis uses G = -8..8.
    # So G_a - G_b can be up to 16.
    # We must compute Fourier coefficients for G in [-16, 16].
    # 32x32 DFT gives us exactly that (indices 0..31 correspond to -16..15 or similar).
    
    # Let's compute the full 32x32 DFT of the 1/eps map.
    # Then map G_diff to the correct DFT index.
    
    # Symmetry orbits
    def get_orbits(size):
        visited = set()
        orbits = []
        for i in range(size):
            for j in range(size):
                if (i, j) not in visited:
                    orbit = set()
                    orbit.add((i, j))
                    orbit.add((size-1-i, size-1-j))
                    orbit.add((i, size-1-j))
                    orbit.add((size-1-i, j))
                    orbit.add((j, i))
                    orbit.add((size-1-j, size-1-i))
                    visited.update(orbit)
                    orbits.append(list(orbit))
        return orbits

    orbits = get_orbits(size)
    num_orbits = len(orbits)
    
    # Precompute orbit pixel lists for fast cell construction
    orbit_pixels = [np.array(o, dtype=int) for o in orbits]
    
    def orbit_to_cell(orbit_vals):
        cell = np.zeros((size, size), dtype=np.uint8)
        for o_idx in range(num_orbits):
            val = orbit_vals[o_idx]
            if val:
                i, j = orbit_pixels[o_idx].T
                cell[i, j] = 1
        return cell

    def compute_fourier_eps_inv(cell):
        """Compute Fourier coefficients of 1/eps from pixel cell.
        Returns array of coefficients for G in [-16, 16]x[-16, 16].
        """
        inv_eps = np.where(cell == 0, 1.0, 1.0/eps_high)
        # 2D DFT
        # numpy fft2: F[kx, ky] = sum_{i,j} f[i,j] exp(-2pi i (kx*i + ky*j)/N)
        # This gives indices 0..N-1.
        # We need indices -16..15 (or -16..16).
        # fftshift moves zero frequency to center.
        F = np.fft.fft2(inv_eps) / (size * size)
        F_shifted = np.fft.fftshift(F)
        # F_shifted[16, 16] corresponds to G=(0,0)
        # G_x = -16..15, G_y = -16..15
        return F_shifted

    # Precompute G_diff indices for the 289x289 Hamiltonian
    # G_vectors are in [-8, 8]
    # G_diff = G_a - G_b is in [-16, 16]
    # We need to map G_diff to index in F_shifted
    # F_shifted index: ix = Gx + 16, iy = Gy + 16
    # But wait, F_shifted covers -16..15.
    # If G_diff = 16, index 32 is out of bounds (size 32, 0..31).
    # G_diff = -16..16.
    # -16 -> 0
    # 0 -> 16
    # 16 -> 32 (wrap around to 0? No, 16 is same as -16 in DFT? Yes, periodicity 32. 16 = -16 mod 32).
    # So index = (G_diff + 16) % 32.
    
    # Precompute the lookup matrix for inv_eps_G
    # lookup[a, b] = index in F_shifted for G_a - G_b
    G_diffs = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (289, 289, 2)
    # G_diffs[..., 0] is Gx_diff, G_diffs[..., 1] is Gy_diff
    # Index in F_shifted: (Gx_diff + 16) % 32, (Gy_diff + 16) % 32
    idx_x = (G_diffs[..., 0] + 16) % 32
    idx_y = (G_diffs[..., 1] + 16) % 32
    # Flatten index into 32x32 array
    lookup_indices = idx_x * 32 + idx_y  # (289, 289)
    
    def compute_bands(inv_eps_F, kx, ky):
        """Compute TM band eigenvalues at given k-point.
        inv_eps_F is the 32x32 shifted FFT array.
        """
        # |k+G|^2
        k_plus_G = G_vectors + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)  # (289,)
        
        # Get inv_eps coefficients for all pairs
        # inv_eps_F is 32x32. Flatten it for indexing.
        inv_eps_flat = inv_eps_F.ravel()
        coeffs = inv_eps_flat[lookup_indices]  # (289, 289)
        
        # H = coeffs * kG2[None, :]
        H = coeffs * kG2[np.newaxis, :]
        
        # H is Hermitian?
        # coeffs[a, b] = (1/eps)_{G_a - G_b}
        # coeffs[b, a] = (1/eps)_{G_b - G_a} = conj(coeffs[a, b])
        # So H is Hermitian.
        # Solve eigenvalue problem
        eigenvalues = np.linalg.eigvalsh(H)
        return eigenvalues

    # K-points for evaluation
    # Irreducible BZ: 0 <= ky <= kx <= 0.5
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.1, 0.0),
        (0.4, 0.1),
        (0.3, 0.3),
        (0.15, 0.05),
        (0.2, 0.1),
        (0.45, 0.2),
        (0.35, 0.35),
        (0.1, 0.05),
        (0.4, 0.4),
    ]
    
    # Precompute kG2 for all k points to save time?
    # kG2 depends on kx, ky.
    # We can precompute the matrix M_k = |k+G|^2 for each k point.
    kG2_cache = []
    for kx, ky in k_points:
        k_plus_G = G_vectors + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)
        kG2_cache.append(kG2)

    def evaluate_cell(orbit_vals):
        """Evaluate the gap-midgap ratio for given orbit values."""
        cell = orbit_to_cell(orbit_vals)
        inv_eps_F = compute_fourier_eps_inv(cell)
        inv_eps_flat = inv_eps_F.ravel()
        
        min_w5_sq = float('inf')
        max_w4_sq = 0.0
        
        for idx_k, (kx, ky) in enumerate(k_points):
            kG2 = kG2_cache[idx_k]
            coeffs = inv_eps_flat[lookup_indices]
            H = coeffs * kG2[np.newaxis, :]
            eigenvalues = np.linalg.eigvalsh(H)
            
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        
        if max_w4_sq <= 0:
            return 0.0
        
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        
        gap = min_w5 - max_w4
        ratio = 2.0 * gap / midgap
        return ratio

    # Optimization
    np.random.seed(42)
    
    best_score = -float('inf')
    best_vals = None
    
    # Initial configs
    initial_configs = []
    
    # 1. Centered circle
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx, orbit in enumerate(orbits):
        i, j = orbit[0]
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        if x*x + y*y < 0.2**2:
            vals[o_idx] = 1
    initial_configs.append(vals.copy())
    
    # 2. Checkerboard
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx, orbit in enumerate(orbits):
        i, j = orbit[0]
        if (i + j) % 2 == 0:
            vals[o_idx] = 1
    initial_configs.append(vals.copy())
    
    # 3. Random
    for _ in range(10):
        vals = np.random.randint(0, 2, num_orbits)
        initial_configs.append(vals.copy())
    
    # Evaluate initial configs
    for config in initial_configs:
        score = evaluate_cell(config)
        if score > best_score:
            best_score = score
            best_vals = config.copy()
    
    # Simulated Annealing
    current_vals = best_vals.copy()
    current_score = best_score
    
    T = 0.5  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    deadline = start_time + 200.0  # Safety margin
    
    while time.time() < deadline:
        # Pick a random orbit to flip
        idx = np.random.randint(num_orbits)
        
        new_vals = current_vals.copy()
        new_vals[idx] = 1 - new_vals[idx]
        
        new_score = evaluate_cell(new_vals)
        
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = new_vals.copy()
        else:
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_vals = new_vals
                current_score = new_score
        
        T = max(T_min, T * cooling)
        
        # Restart if stuck?
        # Simple restart logic: if no improvement for a long time, random restart
        # But with SA, we just let it cool.
        
    # Final local search with single flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(num_orbits):
            new_vals = best_vals.copy()
            new_vals[idx] = 1 - new_vals[idx]
            new_score = evaluate_cell(new_vals)
            if new_score > best_score:
                best_score = new_score
                best_vals = new_vals.copy()
                improved = True
    
    best_cell = orbit_to_cell(best_vals)
    return best_cell.tolist()
