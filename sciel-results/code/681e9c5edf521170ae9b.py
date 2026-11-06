import numpy as np
import time

def design(problem):
    start_time = time.time()
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"] # 1
    eps_high = problem["eps_high"] # 12
    
    # Symmetric cell generation
    def make_cell(radius):
        cell = np.zeros((size, size), dtype=np.int8)
        # Center is at (15.5, 15.5) if 0-indexed? 
        # (i+0.5)/size - 0.5 is in [-0.5, 0.5).
        # Center of pixel (i,j) is (i+0.5)/size - 0.5.
        # Distance from center (0,0) in normalized coords.
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                if x*x + y*y < radius*radius:
                    cell[i, j] = 1
        return cell

    # Fast PWE Solver
    def compute_gap(cell, n_pw=11, k_points=None):
        # n_pw is half-width of plane wave grid, total (2*n_pw+1)^2 waves
        # 11 -> 23x23 = 529 waves. Might be slow. Let's try 7 -> 15x15=225.
        # 7x7 grid of G vectors: Gx, Gy in {-7, ..., 7} * (2*pi/a)
        # Actually, standard is to use a grid of G vectors.
        
        if k_points is None:
            # Sample IBZ: Triangle (0,0) to (pi,0) to (pi,pi)
            # We can use a few points along the edges
            k_points = []
            # Gamma to X
            for i in range(1, 6):
                k_points.append((np.pi * i / 5, 0))
            # X to M
            for i in range(1, 6):
                k_points.append((np.pi, np.pi * i / 5))
            # M to Gamma
            for i in range(1, 6):
                k_points.append((np.pi * (1 - i/5), np.pi * (1 - i/5)))
            k_points.append((0,0))
            k_points.append((np.pi, 0))
            k_points.append((np.pi, np.pi))
            
        # Compute Fourier coefficients of eps_inv
        # eps_inv is 1 where cell=0, 1/eps_high where cell=1
        eps_inv_grid = np.ones((size, size))
        eps_inv_grid[cell == 1] = 1.0 / eps_high
        
        # FFT of eps_inv_grid
        # We need eps_inv(G) for G in the plane wave grid
        # The grid of G vectors is defined by the plane waves used in the solver.
        # Let's define the plane wave grid.
        # G = (m * 2*pi, n * 2*pi) for m,n in -n_max..n_max
        # The FFT of a size N grid gives frequencies 0..N-1.
        # We need to map the FFT output to the G vectors.
        
        n_max = n_pw
        G_vectors = []
        for m in range(-n_max, n_max + 1):
            for n in range(-n_max, n_max + 1):
                G_vectors.append((m * 2 * np.pi, n * 2 * np.pi))
        
        N_g = len(G_vectors)
        
        # Calculate eps_inv(G)
        # The FFT of eps_inv_grid (size 32x32) gives values for frequencies k = 0..31.
        # Frequency k corresponds to G = 2*pi*k/32 = pi*k/16.
        # Our G vectors are integers * 2*pi.
        # So G = 2*pi * m. This corresponds to k = 32 * m / 2 = 16 * m.
        # So we only need FFT components at indices that are multiples of 16?
        # No, the unit cell is 32x32 pixels. The lattice constant is 32 pixels.
        # So the fundamental frequency is 2*pi/32 = pi/16.
        # The G vectors for the reciprocal lattice are integer multiples of 2*pi/a = 2*pi/32 = pi/16?
        # No, reciprocal lattice vector is 2*pi/a. If a=1 (normalized), G = 2*pi * integer.
        # In pixel units, a = 32. G = 2*pi * m / 32.
        # So the G vectors are (2*pi*m/32, 2*pi*n/32).
        # The FFT of the 32x32 array gives coefficients for these exact G vectors!
        # Index (u, v) in FFT corresponds to G = (2*pi*u/32, 2*pi*v/32).
        
        fft_eps_inv = np.fft.fft2(eps_inv_grid)
        
        # We need to map G_vectors to FFT indices.
        # G = (2*pi*m/32, 2*pi*n/32).
        # FFT index u corresponds to Gx = 2*pi*u/32. So u = m.
        # But FFT indices are 0..31. Negative m wrap around.
        # m in -n_max..n_max.
        # Index = m % 32.
        
        eps_inv_G = np.zeros(N_g, dtype=complex)
        for idx, (Gx, Gy) in enumerate(G_vectors):
            # Gx = 2*pi*m/32. m = Gx / (2*pi/32) = Gx * 16 / pi.
            # Since Gx is generated as m*2*pi, this logic holds if we track m,n.
            # Let's regenerate G_vectors with m,n indices to be safe.
            pass
            
        # Regenerate G_vectors with indices
        G_vectors = []
        m_n_list = []
        for m in range(-n_max, n_max + 1):
            for n in range(-n_max, n_max + 1):
                G_vectors.append((m * 2 * np.pi / size, n * 2 * np.pi / size))
                m_n_list.append((m, n))
                
        eps_inv_G = np.zeros(N_g, dtype=complex)
        for idx, (m, n) in enumerate(m_n_list):
            # FFT index
            u = m % size
            v = n % size
            eps_inv_G[idx] = fft_eps_inv[v, u] # Note: numpy fft2 is (row, col) = (y, x)
        
        # Build Matrix M for each k-point
        # M_{ij} = (k+G_i) dot (k+G_j) / eps_inv(G_i - G_j)
        # G_i - G_j = (m_i - m_j) * 2*pi/size, ...
        # We need eps_inv(G_diff).
        # G_diff index in m_n_list?
        # We can precompute a dictionary or array for eps_inv_G_diff.
        
        # Let's precompute eps_inv_G_diff for all differences
        # Actually, eps_inv(G) is periodic in G.
        # We can just look up the value for the difference vector.
        
        # To speed up, we can precompute the matrix for a specific k.
        
        max_band = 5
        min_omega_5 = np.inf
        max_omega_4 = 0
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            
            # Precompute (k+G) vectors
            KG = np.array([k_vec + G for G in G_vectors])
            KG2 = np.sum(KG**2, axis=1) # |k+G|^2
            KG_dot = KG @ KG.T # (k+G_i) dot (k+G_j)
            
            # Construct M
            M = np.zeros((N_g, N_g), dtype=complex)
            
            # We need eps_inv(G_i - G_j)
            # G_i - G_j = (m_i - m_j, n_i - n_j) * 2*pi/size
            # We can compute the FFT coefficient for this difference on the fly?
            # Or precompute a lookup table for all possible differences in our grid.
            # Differences range from -2*n_max to 2*n_max.
            
            # Let's create a 2D array of eps_inv for differences
            # diff_m, diff_n in [-2*n_max, 2*n_max]
            diff_range = 2 * n_max
            eps_inv_diff = np.zeros((2*diff_range+1, 2*diff_range+1), dtype=complex)
            
            # Fill eps_inv_diff from fft_eps_inv
            # fft_eps_inv[u, v] corresponds to G = (2*pi*u/size, 2*pi*v/size)
            # We need G = (2*pi*dm/size, 2*pi*dn/size)
            # So u = dm, v = dn (mod size)
            
            for dm in range(-diff_range, diff_range + 1):
                for dn in range(-diff_range, diff_range + 1):
                    u = dm % size
                    v = dn % size
                    eps_inv_diff[dm + diff_range, dn + diff_range] = fft_eps_inv[v, u]
            
            # Now fill M
            for i in range(N_g):
                mi, ni = m_n_list[i]
                for j in range(N_g):
                    mj, nj = m_n_list[j]
                    dm = mi - mj
                    dn = ni - nj
                    denom = eps_inv_diff[dm + diff_range, dn + diff_range]
                    if denom == 0:
                        M[i, j] = 0
                    else:
                        M[i, j] = KG_dot[i, j] / denom
                        
            # Solve eigenvalue problem
            # M is Hermitian?
            # KG_dot is real symmetric. eps_inv_diff is conjugate symmetric?
            # eps_inv(G) = conj(eps_inv(-G)) for real eps.
            # So M is Hermitian.
            # Use eigh
            try:
                w2, v = np.linalg.eigh(M)
            except np.linalg.LinAlgError:
                continue
                
            # w2 are eigenvalues (omega^2)
            # We want bands 4 and 5 (index 3 and 4)
            if len(w2) >= 5:
                omega_4 = np.sqrt(w2[3])
                omega_5 = np.sqrt(w2[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
                
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        if midgap == 0:
            return 0
        return gap / midgap

    # Search for best radius
    best_gap = -np.inf
    best_cell = None
    
    # Coarse search
    radii = np.linspace(0.1, 0.45, 20)
    for r in radii:
        if time.time() - start_time > 100:
            break
        cell = make_cell(r)
        gap_ratio = compute_gap(cell, n_pw=7) # 15x15 waves
        if gap_ratio > best_gap:
            best_gap = gap_ratio
            best_cell = cell
            best_radius = r
            
    # Refine search around best_radius
    if best_cell is not None:
        radii_refine = np.linspace(best_radius - 0.05, best_radius + 0.05, 10)
        for r in radii_refine:
            if time.time() - start_time > 115:
                break
            cell = make_cell(r)
            gap_ratio = compute_gap(cell, n_pw=9) # 19x19 waves for better accuracy
            if gap_ratio > best_gap:
                best_gap = gap_ratio
                best_cell = cell
                best_radius = r
                
    # Convert to list of lists
    if best_cell is None:
        best_cell = make_cell(0.2)
        
    return best_cell.tolist()
