import time
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # Safe margin before 120s limit
    
    size = problem["pixels"] # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Define the Surrogate Solver Parameters
    # We use a smaller number of plane waves for speed.
    # The exact evaluator uses 289 (17x17). We use 9 (3x3) or 25 (5x5).
    # 5x5 is a good balance of speed and accuracy for a surrogate.
    num_pw = 5  # 5x5 = 25 plane waves
    pw_indices = np.array([i for i in range(-num_pw//2, num_pw//2 + 1)])
    G = np.stack(np.meshgrid(pw_indices, pw_indices, indexing='ij'), axis=-1)
    G = G.reshape(-1, 2).astype(float)
    
    # Coarse grid for fast Fourier transform of the dielectric profile
    # The full grid is 32x32. We can use the full grid if FFT is fast enough,
    # but for gradient descent, a slightly coarser representation or just 
    # using the full grid with vectorized operations is best.
    # Let's stick to the full 32x32 grid for accuracy in the surrogate, 
    # but optimize a 16x16 fundamental domain (1/4 symmetry) to reduce search space.
    
    # Symmetry Strategy:
    # The cell is invariant under D4 (rotations and reflections).
    # We can define the cell by the top-left quadrant (0..15, 0..15) 
    # and enforce:
    # cell[i, j] == cell[-i, -j] (inversion/rotation)
    # cell[i, j] == cell[j, i] (reflection across y=x)
    # Actually, simpler: 
    # cell[i,j] = cell[size-1-i, size-1-j]
    # cell[i,j] = cell[j, i]
    # cell[i,j] = cell[size-1-i, j] (reflection across horizontal midline)
    
    # Let's optimize a 16x16 quadrant (top-left) and mirror it.
    # Indices 0..15.
    # Mirroring:
    # col 16..31 is mirror of 15..0
    # row 16..31 is mirror of 15..0
    # Transpose also applies.
    # The fundamental domain for D4 symmetry in a 32x32 grid centered at (16,16)
    # is actually a bit tricky to map perfectly with simple 16x16 blocks due to the center line.
    # However, a simpler symmetry constraint often used is:
    # cell[i,j] == cell[j,i] and cell[i,j] == cell[N-1-i, N-1-j]
    # This ensures 180 deg rotation and reflection.
    # Let's enforce:
    # 1. Transpose symmetry: C = C^T
    # 2. 180 Rotation: C[i,j] = C[N-1-i, N-1-j]
    # This is sufficient for the "square symmetry group" mentioned if we assume the evaluator checks these.
    # The prompt says: "invariant under the square symmetry group ... cell == cell[::-1] row-wise and column-wise and cell == its transpose"
    # This implies:
    # C[i,j] = C[N-1-i, j] (Row reversal)
    # C[i,j] = C[i, N-1-j] (Col reversal)
    # C[i,j] = C[j, i]     (Transpose)
    
    # We can optimize the top-left quadrant (0..15, 0..15) but we must be careful with the center lines.
    # For N=32, indices 0..31.
    # Row 15 and 16 are the center lines.
    # If we enforce C[i,j] = C[31-i, 31-j] and C[i,j]=C[j,i], it covers the group.
    
    # Let's optimize a 16x16 domain (indices 0..15) and reflect.
    # But wait, if we just reflect 0..15 to 16..31, do we satisfy all conditions?
    # Let d[i,j] be the 16x16 block.
    # Full cell C:
    # C[0:16, 0:16] = d
    # C[0:16, 16:32] = d[:, ::-1]  # Col reflection
    # C[16:32, 0:16] = d[::-1, :]  # Row reflection
    # C[16:32, 16:32] = d[::-1, ::-1]
    
    # Check Transpose:
    # C[i,j] = C[j,i]?
    # If i,j < 16: C[i,j]=d[i,j], C[j,i]=d[j,i]. We need d to be symmetric d[i,j]=d[j,i].
    # If i<16, j>=16: C[i,j] = d[i, 31-j]. C[j,i] = C[31-j, i].
    # j>=16 -> 31-j < 16. Let j' = 31-j.
    # C[i, 31-j'] = d[i, j'].
    # C[31-j', i] is in the bottom-left block (row >=16, col <16).
    # C[16.., 0..] is d[::-1, :]. Index 31-j' in row means d[15-(31-j')-? No.
    # Let's use 0-indexed.
    # Row 16..31 corresponds to d[15..0].
    # Row index r in C (16<=r<=31) maps to d[31-r].
    # Col index c in C (0<=c<=15) maps to d[c].
    # So C[r, c] = d[31-r, c].
    # We need C[i, j] = C[j, i].
    # Case i < 16, j >= 16.
    # C[i, j] = d[i, 31-j].
    # C[j, i]: j is row, i is col. j>=16, i<16.
    # C[j, i] = d[31-j, i].
    # So we need d[i, 31-j] = d[31-j, i].
    # Let u = i, v = 31-j. Since j>=16, v <= 15.
    # We need d[u, v] = d[v, u].
    # This is just symmetry of d.
    # So, if we optimize a 16x16 symmetric matrix d (d[i,j]=d[j,i]) and construct C as above, it satisfies all symmetries.
    
    # Optimization Variable: 16x16 symmetric density matrix d (256 unique params -> 136 due to symmetry? No, just optimize full 16x16 and symmetrize).
    # Actually, simpler: Optimize 16x16 array `d_quad`.
    
    def construct_cell(d_quad):
        """Construct 32x32 cell from 16x16 symmetric quadrant."""
        C = np.zeros((size, size))
        C[:16, :16] = d_quad
        C[:16, 16:] = d_quad[:, ::-1]
        C[16:, :16] = d_quad[::-1, :]
        C[16:, 16:] = d_quad[::-1, ::-1]
        return C

    # Surrogate Band Solver
    # Uses the 25 plane waves defined by G.
    # We need the Fourier coefficients of epsilon(r) at each G vector.
    # epsilon(r) = eps_low + (eps_high - eps_low) * density(r)
    # eps_G = (1/N^2) * sum_{i,j} eps(r_{i,j}) * exp(-i G . r_{i,j})
    # r_{i,j} = (i+0.5, j+0.5) in units of cell size L=1.
    
    # Precompute positions
    x = np.arange(size) + 0.5
    y = np.arange(size) + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.stack([X, Y], axis=-1) # (32, 32, 2)
    
    # G vectors are in units of 2*pi/L.
    # Phase = G . R
    # G is (M, 2). R is (32, 32, 2).
    # We want phase[i, j, k] = G[k, 0]*X[i, j] + G[k, 1]*Y[i, j]
    
    # For speed in the loop, we can precompute the complex exponentials for the density basis.
    # But density changes.
    # eps_G = eps_avg + delta_eps * (1/N^2) sum density[i,j] exp(-i G_k . R_ij)
    # Let F_k = (1/N^2) sum density[i,j] exp(-i G_k . R_ij)
    # This is an FFT of the density (with half-pixel shift).
    
    # Since we use a small grid of G vectors (5x5), we can just sum.
    # M = 25. N^2 = 1024.
    # 25 * 1024 operations per evaluation. Very fast.
    
    def compute_fourier_coeffs(density):
        """
        density: 32x32 array
        Returns: eps_G for the 25 plane waves
        """
        # eps_avg is the average epsilon
        eps_avg = np.mean(eps_low + (eps_high - eps_low) * density)
        
        # Compute exp(-i G . R)
        # G is (25, 2)
        # R is (32, 32, 2)
        # We can reshape R to (1024, 2)
        R_flat = R.reshape(-1, 2)
        
        # Phase matrix (1024, 25)
        # Phase[i, k] = R_flat[i, 0]*G[k, 0] + R_flat[i, 1]*G[k, 1]
        phase = R_flat @ G.T
        
        # Exp matrix (1024, 25)
        exp_phase = np.exp(-1j * phase)
        
        # density_flat (1024, 1)
        d_flat = density.reshape(-1, 1)
        
        # F_k = (1/N^2) * sum(d_flat * exp_phase[:, k])
        F = (d_flat.T @ exp_phase).T / (size * size)
        
        # eps_G = eps_avg + (eps_high - eps_low) * F
        # Note: For TM mode, the matrix is (eps^{-1} * (G+K)^2)
        # Wait, the standard PW method for TM uses:
        # [ (G+K)^2 / eps(G) ] c_{G'} = omega^2 c_G
        # So we need 1/eps(G).
        
        eps_G = eps_avg + (eps_high - eps_low) * F
        return 1.0 / eps_G, eps_G # Return inverse for TM

    def solve_bands_surrogate(density, k_point):
        """
        Solve for the first 5 bands at a specific k point.
        density: 32x32 array
        k_point: (kx, ky) in units of 2*pi/L
        """
        inv_eps_G, _ = compute_fourier_coeffs(density)
        
        # Build the matrix M
        # M_{G, G'} = (G + K)_x^2 + (G + K)_y^2 ? No.
        # The equation is:
        # sum_{G'} [ (G+K) . (G'+K) / eps(G-G') ] c_{G'} = omega^2 c_G ?
        # Standard form:
        # (G+K)^2 c_G + sum_{G' != 0} [ (G+K).G' / eps(G') ] c_{G-G'} = omega^2 c_G
        # Or in matrix form A c = omega^2 c:
        # A_{G, G'} = delta_{G,G'} * (G+K)^2 + (1 - delta_{G,G'}) * (G+K).G' / eps(G-G')
        
        # Let's construct A (25x25)
        # G is (25, 2)
        # G_plus_K = G + K
        GpK = G + np.array(k_point)
        
        # Squared norms of G+K
        GpK_sq = np.sum(GpK**2, axis=1)
        
        A = np.zeros((len(G), len(G)), dtype=complex)
        
        # Precompute dot products (G+K) . G'
        # This is GpK @ G.T
        dot_products = GpK @ G.T
        
        for i in range(len(G)):
            for j in range(len(G)):
                Gi = G[i]
                Gj = G[j]
                diff = Gi - Gj
                # Check if diff is one of our G vectors?
                # Our G vectors are defined by integer indices.
                # diff is a vector. We need to find which index k corresponds to diff.
                # We can use a dictionary or just loop through G to find the match.
                # Since M is small (25), linear search is fine.
                idx_diff = -1
                for k in range(len(G)):
                    if np.allclose(G[k], diff, atol=1e-5):
                        idx_diff = k
                        break
                
                if i == j:
                    A[i, j] = GpK_sq[i]
                else:
                    if idx_diff == -1:
                        # If diff is not in our set, the coupling is 0 in this truncated basis
                        # (This is an approximation, but standard for small N)
                        A[i, j] = 0.0
                    else:
                        # A[i, j] = dot(Gi+K, Gj) / eps(Gi-Gj)
                        # Note: The standard formula usually has (G+K) dot (G'+K) in the diagonal 
                        # and (G+K) dot G' in off-diagonal?
                        # Let's check the exact form.
                        # omega^2 c_G = (G+K)^2 c_G + sum_{G'!=0} ((G+K) . G') / eps(G') c_{G-G'}
                        # Here G' is the difference.
                        # So A[G, G''] where G'' = G - G'
                        # A[i, j] corresponds to c_j coupling to c_i.
                        # c_i is associated with G_i.
                        # Term in sum for c_i: sum_{G'} ... c_{G_i - G'}
                        # So we need c_j such that j = i - G' -> G' = G_i - G_j.
                        # Coupling coefficient: ((G_i + K) . G') / eps(G')
                        # G' = G_i - G_j.
                        # Coeff = ((G_i + K) . (G_i - G_j)) / eps(G_i - G_j)
                        
                        # Wait, the standard PW expansion is:
                        # E = sum c_{G} exp(i (G+K) . r)
                        # Laplacian gives (G+K)^2.
                        # 1/eps(r) expands to sum eps_{-G'} exp(i G' . r)
                        # Product: sum_{G', G''} c_{G''} eps_{G'} exp(i (G''+G'+K) . r)
                        # We want this to equal omega^2 sum c_G exp(i (G+K) . r)
                        # So G = G'' + G' -> G'' = G - G'
                        # Coeff for c_{G-G'} in the equation for c_G is eps_{G'}
                        # Equation: (G+K)^2 c_G + sum_{G'!=0} ( (G+K) . G' / eps(G') ) c_{G-G'} = omega^2 c_G ??
                        # No, it's:
                        # (G+K)^2 c_G + sum_{G'!=0} [ ( (G+K) . G' ) / eps(G') ] c_{G-G'} = omega^2 c_G
                        # This looks correct for TM.
                        
                        # So A[i, j] (coeff of c_j in row i)
                        # j = i - G' -> G' = G_i - G_j
                        # A[i, j] = ( (G_i + K) . (G_i - G_j) ) / eps(G_i - G_j)
                        
                        diff = G[i] - G[j]
                        # We already found idx_diff for diff.
                        eps_diff = inv_eps_G[idx_diff]
                        dot_term = np.dot(G[i] + np.array(k_point), diff)
                        A[i, j] = dot_term * eps_diff

        # Diagonal is (G+K)^2
        for i in range(len(G)):
             A[i, i] = GpK_sq[i]
             
        # Solve Eigenvalues
        # A is Hermitian (if eps is real). 
        # np.linalg.eigvalsh
        try:
            eigvals = np.linalg.eigvalsh(A)
        except:
            return []
            
        # Return first 5
        return sorted(eigvals[:5].real)

    # Optimization Loop
    # We optimize `d_quad` (16x16).
    # Initial guess: A dielectric hole in the center of the quadrant? 
    # Or just 0.5 everywhere?
    # Let's start with a pattern that resembles a high gap cell.
    # A central rod is good. 
    # In the 16x16 quadrant, the cell center is at (15.5, 15.5) roughly?
    # No, the full cell center is (16, 16).
    # The quadrant covers (0..15, 0..15).
    # The corner (15,15) is the center of the cell.
    # So we want high density at (15,15) and maybe lower density at (0,0)?
    # Typical high gap: High density at center, low at corners? Or vice versa?
    # For TM, usually high dielectric in the center of the cell (where field is high/low depending on band).
    # Let's start with a smooth gradient.
    
    d_quad = np.ones((16, 16)) * 0.5
    # Add a central peak at (15,15)
    d_quad[13:16, 13:16] = 0.9
    
    # Symmetrize d_quad
    d_quad = (d_quad + d_quad.T) / 2.0

    # K points to evaluate
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 1
    # Boundary: (0,0) to (0.5, 0.5) to (1, 0) to (1,1)? No.
    # Standard irreducible: Triangle (0,0)-(0.5,0.5)-(1,0) ? No.
    # Square lattice: (0,0) to (0.5, 0.5) is X-M-X path?
    # Gamma (0,0) to X (0.5, 0) ? No, X is (1,0) or (0,1) depending on convention.
    # Usually Gamma (0,0), X (1,0), M (1,1) in units of 2pi/L?
    # If G vectors are integers, reciprocal lattice vectors are (1,0), (0,1).
    # BZ boundaries: kx=0.5, ky=0.5.
    # Irreducible: 0 <= kx <= 0.5, 0 <= ky <= kx ? (If 90 deg symmetry)
    # Or 0 <= kx <= ky <= 0.5 ?
    # Let's sample points in the triangle (0,0), (0.5,0), (0.5,0.5).
    
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.25, 0.0),
        (0.5, 0.0),      # X
        (0.25, 0.25),
        (0.5, 0.25),
        (0.5, 0.5),      # M
        (0.1, 0.05),
        (0.3, 0.1),
        (0.4, 0.2),
        (0.45, 0.45),
        (0.2, 0.3),
        (0.3, 0.4),
    ]

    def evaluate_score(d_quad):
        cell = construct_cell(d_quad)
        # Threshold to 0/1 for the surrogate? 
        # The evaluator uses 0/1. The surrogate can use continuous to allow gradients.
        # But the "exact Fourier coefficients of the pixel map" implies discrete.
        # However, for optimization, using continuous density in the surrogate 
        # allows us to navigate the space. We binarize at the end.
        # But binarization creates discontinuities.
        # A common trick: Use a sigmoid or just optimize continuous and binarize.
        # Let's optimize continuous density, but the surrogate solver should ideally
        # reflect the discrete nature.
        # Given time constraints, we will optimize continuous density with the surrogate,
        # then binarize, then maybe do a few local pixel flips.
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for kp in k_points:
            omegas = solve_bands_surrogate(cell, kp)
            if len(omegas) >= 5:
                max_omega_4 = max(max_omega_4, omegas[3])
                min_omega_5 = min(min_omega_5, omegas[4])
                
        if min_omega_5 == np.inf:
            return 0.0, max_omega_4, min_omega_5
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0, max_omega_4, min_omega_5
            
        ratio = 2 * gap / midgap
        return ratio, max_omega_4, min_omega_5

    # Simple Gradient Descent / Hill Climbing
    # Since we don't have easy gradients for the min/max of eigenvalues,
    # we use a stochastic approach or finite differences on a few key points.
    
    best_d_quad = d_quad.copy()
    best_score, _, _ = evaluate_score(best_d_quad)
    
    # Optimize
    step_size = 0.1
    num_iterations = 200
    
    for i in range(num_iterations):
        if time.time() > deadline - 20:
            break
            
        # Random perturbation in a small region
        # Pick a random 2x2 or 3x3 block in d_quad
        r0 = np.random.randint(0, 13)
        c0 = np.random.randint(0, 13)
        delta = np.random.uniform(-step_size, step_size)
        
        d_trial = best_d_quad.copy()
        d_trial[r0:r0+3, c0:c0+3] += delta
        d_trial = np.clip(d_trial, 0, 1)
        d_trial = (d_trial + d_trial.T) / 2.0
        
        score, _, _ = evaluate_score(d_trial)
        
        if score > best_score:
            best_score = score
            best_d_quad = d_trial.copy()
            # Decrease step size
            step_size *= 0.95
        else:
            step_size *= 0.99

    # Final Binarization and Local Refinement
    # Convert continuous density to 0/1
    # Threshold 0.5
    d_bin = (best_d_quad > 0.5).astype(float)
    # Symmetrize binary
    d_bin = (d_bin + d_bin.T) / 2.0
    d_bin = (d_bin > 0.5).astype(float)
    
    cell_bin = construct_cell(d_bin)
    
    # Local Pixel Flip Search
    # Try flipping pixels in the full 32x32 cell that respect symmetry
    # We can flip a pixel and its 3 symmetric counterparts.
    
    def get_symmetric_partners(i, j):
        parts = set()
        parts.add((i, j))
        parts.add((j, i))
        parts.add((31-i, 31-j))
        parts.add((31-j, 31-i))
        return list(parts)

    def evaluate_cell_score(cell):
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        for kp in k_points:
            omegas = solve_bands_surrogate(cell, kp)
            if len(omegas) >= 5:
                max_omega_4 = max(max_omega_4, omegas[3])
                min_omega_5 = min(min_omega_5, omegas[4])
        if min_omega_5 == np.inf:
            return 0.0
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return 2 * gap / midgap

    best_cell = cell_bin
    best_final_score = evaluate_cell_score(best_cell)
    
    # Refinement: Try flipping unique pixels in the 16x16 quadrant
    # If we flip (i,j) in d_bin, it flips the symmetric partners in the full cell.
    
    for i in range(20): # 20 refinement steps
        if time.time() > deadline - 5:
            break
        improved = False
        for r in range(16):
            for c in range(r+1, 16): # Upper triangle
                # Try flipping
                d_trial = best_d_quad.copy() # Use continuous? No, use binary d_bin
                # Let's operate on d_bin directly
                d_trial_bin = (best_d_quad > 0.5).astype(float)
                d_trial_bin = (d_trial_bin + d_trial_bin.T) / 2.0
                d_trial_bin = (d_trial_bin > 0.5).astype(float)
                
                # Flip
                if d_trial_bin[r, c] == 1:
                    d_trial_bin[r, c] = 0
                    d_trial_bin[c, r] = 0
                else:
                    d_trial_bin[r, c] = 1
                    d_trial_bin[c, r] = 1
                    
                d_trial_bin = (d_trial_bin + d_trial_bin.T) / 2.0
                d_trial_bin = (d_trial_bin > 0.5).astype(float)
                
                cell_trial = construct_cell(d_trial_bin)
                score = evaluate_cell_score(cell_trial)
                
                if score > best_final_score:
                    best_final_score = score
                    best_cell = cell_trial
                    best_d_quad = d_trial_bin
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Return the best cell
    # Ensure 0/1 ints
    return best_cell.astype(int).tolist()

