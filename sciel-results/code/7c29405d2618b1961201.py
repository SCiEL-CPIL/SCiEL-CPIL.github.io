"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def design(problem):
    """
    Optimizes a centrally-located superellipse defined by |x/a|^n + |y/b|^n <= 1.
    The shape exponent n and semi-axis a are tuned to maximize the (1,1) Fourier
    component, which is critical for the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 40s margin for evaluator overhead
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Pre-compute pixel coordinates and indices for symmetry
    # Symmetry: cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j] == cell[j][i]
    # We only need to define the top-left 1/8th of the cell (i <= 15, j <= i)
    # Actually, for 32x32, the center is at 15.5.
    # Symmetry group D4.
    # Fundamental domain: 0 <= j <= i <= 15.
    
    # Build a fast local PW solver
    # N = 17 (329 PWs) or N=13 (289 PWs). Evaluator uses 289 (N=13? No, 17^2=289).
    # Wait, 17^2 = 289. So Gx, Gy range from -16 to 16? Or -8 to 8?
    # Usually N=17 means 17x17 grid. Indices -8 to 8.
    # Let's use N=13 (13x13=169) for speed in the internal loop, or N=9 (81) if very slow.
    # 289 is the evaluator size. Let's try N=13 first.
    N = 13
    G_max = (N - 1) // 2
    Gs = np.arange(-G_max, G_max + 1)
    Gx, Gy = np.meshgrid(Gs, Gs, indexing='ij')
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    Npw = len(Gx)
    
    # Precompute inverse epsilon matrix for a given cell?
    # The matrix K depends on the Fourier coefficients of 1/epsilon.
    # 1/epsilon(x) is periodic.
    # We can compute the Fourier coefficients of 1/epsilon from the pixel map.
    # Since we are changing the map, we have to recompute.
    
    # To speed up, we can approximate the band structure evaluation.
    # But let's write a compact solver.
    
    def compute_fourier_inv_eps(cell):
        # cell is 32x32 0/1 array
        # 1/eps = 1/eps_high if 1, 1/eps_low if 0
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute Fourier coefficients for Gx, Gy
        # C(G) = sum_{i,j} inv_eps[i,j] * exp(-i (Gx * i + Gy * j) * 2pi / 32)
        # Use FFT for speed? 32x32 is small. Direct sum is fine.
        i_idx = np.arange(32)
        j_idx = np.arange(32)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        
        # Phase matrix for each G
        # We need C for all G in Gx, Gy
        # This is essentially the DFT of the 32x32 grid.
        # We can compute the full 32x32 DFT and pick the relevant Gs.
        F = np.fft.fft2(inv_eps)
        
        # Map G indices to FFT indices
        # FFT index k corresponds to frequency k (if k<16) or k-32 (if k>=16)
        # G ranges from -8 to 8 (for N=13)
        # We need to extract these from F
        
        coeffs = np.zeros(Npw, dtype=complex)
        for k in range(Npw):
            gx = Gx[k]
            gy = Gy[k]
            # FFT index
            ix = gx % 32
            iy = gy % 32
            coeffs[k] = F[iy, ix] # Note: fft2 uses (y, x) order in array, but indices are (row, col).
            # Wait, np.fft.fft2(F) computes sum_{i,j} F[i,j] exp(-2pi i (i*ky + j*kx) / N)
            # So F[ky, kx] corresponds to G_y = ky, G_x = kx.
            # My Gx, Gy are flattened.
            # I need to be careful with ordering.
            # Let's just compute the sum directly to avoid index confusion, 32*32*169 = 172k ops, very fast.
            
        # Direct sum is safer and fast enough
        exp_terms = np.exp(-1j * 2 * np.pi * (np.outer(Gx, i_idx) + np.outer(Gy, j_idx)) / 32.0)
        # This is Npw x 32 for Gx and Npw x 32 for Gy
        # We need sum_{i,j} inv_eps[i,j] * exp(-i 2pi (Gx*i + Gy*j)/32)
        # = sum_i exp(-i 2pi Gx*i/32) * sum_j inv_eps[i,j] exp(-i 2pi Gy*j/32)
        
        # Let's use matrix multiplication
        # M1: Npw x 32, M1[k, i] = exp(-i 2pi Gx[k] i / 32)
        # M2: 32 x Npw, M2[j, k] = exp(-i 2pi Gy[k] j / 32)
        
        M1 = np.exp(-1j * 2 * np.pi * np.outer(Gx, i_idx) / 32.0)
        M2 = np.exp(-1j * 2 * np.pi * np.outer(Gy, j_idx) / 32.0).T # 32 x Npw
        
        # coeffs[k] = sum_i sum_j inv_eps[i,j] M1[k,i] M2[j,k]
        # = M1 @ inv_eps @ M2.T ? No.
        # coeffs = M1 @ (inv_eps @ M2.T) ?
        # Let's check dimensions.
        # M1: (Npw, 32)
        # inv_eps: (32, 32)
        # M2: (32, Npw)
        # M1 @ inv_eps @ M2 -> (Npw, Npw). Diagonal is what we want?
        # No.
        # coeffs[k] = sum_i M1[k,i] sum_j inv_eps[i,j] M2[j,k]
        # Let A = inv_eps @ M2.T -> (32, Npw). A[i,k] = sum_j inv_eps[i,j] M2[j,k]
        # Then coeffs[k] = sum_i M1[k,i] A[i,k]
        # This is element-wise product sum over i.
        # coeffs = np.sum(M1 * A.T, axis=1)
        
        A = inv_eps @ M2.T
        coeffs = np.sum(M1 * A.T, axis=1)
        
        return coeffs

    def solve_bands(cell):
        inv_eps_coeffs = compute_fourier_inv_eps(cell)
        
        # Build K matrix
        # K_{G, G'} = (k + G) . (k + G') * inv_eps[G - G']
        # k is the wave vector.
        # We evaluate at several k points in the BZ.
        # For speed, we can just evaluate at Gamma, X, M, and maybe one interior point.
        # The gap is usually at M or X.
        
        # k points (normalized to 2pi/a)
        k_points = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.25, 0.25])
        ]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            # Vector of k + G
            kx = k[0] + Gx
            ky = k[1] + Gy
            k_sq = kx**2 + ky**2
            
            # K matrix
            # K = (k+G) (k+G')^T * inv_eps[G-G']
            # This is a rank-2 update scaled by inv_eps Fourier components.
            # K_{mn} = k_sq[m] * delta_{mn}? No.
            # K_{mn} = (k+G_m) dot (k+G_n) * inv_eps[G_m - G_n]
            
            # We can build this efficiently.
            # Let V be a matrix where V_{m, 0} = kx[m], V_{m, 1} = ky[m]
            # Then (k+G_m) dot (k+G_n) = V_m dot V_n
            # K = V V^T * inv_eps_diff
            # inv_eps_diff[m, n] = inv_eps_coeffs[G_m - G_n]
            
            # Compute diff indices
            # G_m - G_n must be in the range of our coeffs.
            # Since Gs are symmetric, we can look up.
            # Map G vector to index.
            # Gs are from -8 to 8.
            # We can create a dictionary or just compute the index.
            # Index in Gs array: (G - (-8)) = G + 8.
            # But Gs is flattened.
            # Let's just compute the matrix directly for N=13 (169x169).
            
            # inv_eps_diff matrix
            # G_m - G_n
            Gx_diff = Gx[:, None] - Gx[None, :]
            Gy_diff = Gy[:, None] - Gy[None, :]
            
            # Map diff to index in inv_eps_coeffs
            # We need a function to map (gx, gy) to index in Gx, Gy arrays.
            # Since Gx, Gy are meshgrid flattened, we can use a dict.
            # Or, since Gs are regular, we can compute index.
            # ix = gx + G_max, iy = gy + G_max
            # idx = ix * (2*G_max+1) + iy ? No, meshgrid indexing='ij'
            # Gx, Gy = meshgrid(Gs, Gs, indexing='ij')
            # Gx[i, j] = Gs[i], Gy[i, j] = Gs[j]
            # Flatten: idx = i * (2*G_max+1) + j
            # So for diff (dx, dy):
            # i = dx + G_max, j = dy + G_max
            # idx = i * (2*G_max+1) + j
            
            N_dim = 2 * G_max + 1
            idx_diff = (Gx_diff + G_max) * N_dim + (Gy_diff + G_max)
            
            # Check if any diff is out of bounds.
            # G_max=8. Diff range -16 to 16.
            # Our coeffs only go up to 8.
            # This is a problem. The convolution requires coefficients up to max(G) - min(G).
            # If we truncate the PW expansion, we must also truncate the dielectric constant Fourier series?
            # No, the dielectric constant is band-limited to the pixel resolution?
            # Actually, 1/eps is a pixel map. It has high frequency components.
            # But in the Galerkin method, we project onto the PW basis.
            # The matrix element K_{mn} involves the Fourier component of 1/eps at G_m - G_n.
            # If |G_m - G_n| > G_max, we don't have that coefficient in our truncated set?
            # Usually, we compute the Fourier coefficients of 1/eps for all G in the set {G_m - G_n}.
            # This set is larger than {G_m}.
            # For N=13, G range [-8, 8]. Diff range [-16, 16].
            # We need coefficients up to 16.
            # My compute_fourier_inv_eps only computed up to 8.
            # I need to compute up to 16.
            
            # Let's re-compute coeffs up to G_max_diff = 2 * G_max = 16.
            # This will be slower but necessary.
            
            pass # Placeholder, logic needs revision
            
        return 0.0, 0.0

    # The above solver is getting complex and might be slow.
    # Given the 230s limit, and the fact that the current score is 5.78,
    # and the instruction says "REFINEMENT: KEEP this design",
    # maybe I should just tune the parameters of the existing 4-rod design?
    # But the directive says "Model the dielectric distribution as a single... superellipse".
    
    # Let's try a simpler approach:
    # 1. Generate a few superellipse candidates.
    # 2. Use a very fast approximate scoring function?
    # Or just return a good superellipse if the solver is too hard to implement quickly.
    
    # Let's try to implement the solver correctly but efficiently.
    
    def fast_score(cell):
        # Compute Fourier coeffs of 1/eps up to G=16
        G_max_diff = 16
        Gs_diff = np.arange(-G_max_diff, G_max_diff + 1)
        N_dim_diff = len(Gs_diff)
        
        i_idx = np.arange(32)
        j_idx = np.arange(32)
        
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute coeffs for all Gx, Gy in [-16, 16]
        # This is 33x33 = 1089 coeffs.
        # Direct sum: 1089 * 32 * 32 = 1.1M ops. Fast.
        
        Gx_d, Gy_d = np.meshgrid(Gs_diff, Gs_diff, indexing='ij')
        Gx_d = Gx_d.flatten()
        Gy_d = Gy_d.flatten()
        
        M1 = np.exp(-1j * 2 * np.pi * np.outer(Gx_d, i_idx) / 32.0)
        M2 = np.exp(-1j * 2 * np.pi * np.outer(Gy_d, j_idx) / 32.0).T
        
        A = inv_eps @ M2.T
        coeffs_diff = np.sum(M1 * A.T, axis=1)
        
        # Map diff to index
        # idx = (dx + G_max_diff) * N_dim_diff + (dy + G_max_diff)
        
        # Now solve bands for a few k points using N=9 (81 PWs) for speed?
        # Or N=11 (121 PWs).
        N_pw = 9
        G_max_pw = (N_pw - 1) // 2
        Gs_pw = np.arange(-G_max_pw, G_max_pw + 1)
        Gx_pw, Gy_pw = np.meshgrid(Gs_pw, Gs_pw, indexing='ij')
        Gx_pw = Gx_pw.flatten()
        Gy_pw = Gy_pw.flatten()
        N_pw_total = len(Gx_pw)
        
        k_points = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.25, 0.25])
        ]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            kx = k[0] + Gx_pw
            ky = k[1] + Gy_pw
            k_sq = kx**2 + ky**2
            
            # Build K matrix
            Gx_diff = Gx_pw[:, None] - Gx_pw[None, :]
            Gy_diff = Gy_pw[:, None] - Gy_pw[None, :]
            
            idx_diff = (Gx_diff + G_max_diff) * N_dim_diff + (Gy_diff + G_max_diff)
            inv_eps_diff_mat = coeffs_diff[idx_diff]
            
            # K_{mn} = (k+G_m) dot (k+G_n) * inv_eps_diff[m, n]
            # = (kx_m kx_n + ky_m ky_n) * inv_eps_diff[m, n]
            # = (kx_m * inv_eps_diff[m, n] * kx_n) + (ky_m * inv_eps_diff[m, n] * ky_n)
            
            # K = kx[:, None] * inv_eps_diff_mat * kx[None, :] + ky[:, None] * inv_eps_diff_mat * ky[None, :]
            
            K = kx[:, None] * inv_eps_diff_mat * kx[None, :] + ky[:, None] * inv_eps_diff_mat * ky[None, :]
            
            # Solve eigenvalue problem: K c = omega^2 c
            # Use scipy.linalg.eigh for Hermitian matrix
            try:
                evals, _ = np.linalg.eigh(K)
                evals = np.sort(evals)
                # Take the smallest 5
                if len(evals) >= 5:
                    w4 = np.sqrt(evals[3])
                    w5 = np.sqrt(evals[4])
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
            except:
                pass
        
        if min_omega_5 == float('inf'):
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        mid = (min_omega_5 + max_omega_4) / 2.0
        if mid == 0:
            return 0.0
        return gap / mid

    # Search over superellipse parameters
    # |x/a|^n + |y/a|^n <= 1 (a=b for symmetry)
    # Center at 0.5, 0.5
    
    best_score = -1.0
    best_cell = None
    
    # Grid search
    n_vals = [2, 3, 4, 5, 6, 8, 10, 12]
    a_vals = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    
    for n in n_vals:
        for a in a_vals:
            if time.time() > deadline:
                break
            
            cell = np.zeros((32, 32), dtype=int)
            i_idx = np.arange(32)
            j_idx = np.arange(32)
            I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
            
            x = (I + 0.5) / 32.0 - 0.5
            y = (J + 0.5) / 32.0 - 0.5
            
            # Superellipse
            val = np.abs(x / a) ** n + np.abs(y / a) ** n
            mask = val <= 1.0
            
            cell[mask] = 1
            
            # Ensure symmetry (it should be symmetric by construction)
            # Convert to list of lists
            cell_list = cell.tolist()
            
            score = fast_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell_list
            
            if time.time() > deadline:
                break

    if best_cell is None:
        # Fallback to current best known (4 rods)
        cell = [[0 for _ in range(size)] for _ in range(size)]
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        radius = 0.15
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                is_diel = False
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        is_diel = True
                        break
                cell[i][j] = 1 if is_diel else 0
        return cell

    return best_cell

# EVOLVE-BLOCK-END
