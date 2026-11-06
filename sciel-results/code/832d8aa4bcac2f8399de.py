"""
Strategy: Local optimization of a structured photonic crystal unit cell.
We define a parametric family of symmetric designs (central disk + 4 symmetric arms/crosses)
and use a fast FFT-based plane-wave approximation to evaluate the band gap.
We then perform a local search (hill climbing) to maximize the gap between TM bands 4 and 5.
"""
import time
import numpy as np
import sys

def _make_symmetric(cell):
    """
    Enforce 4-fold rotational and reflection symmetry about the center.
    The cell must satisfy:
    cell == cell[::-1] (row flip)
    cell == cell[::-1, ::-1] (col flip) -- actually cell[::-1] covers row flip, cell.T covers transpose?
    The problem states: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    Wait, "cell == cell[::-1] row-wise" means cell[i][j] == cell[N-1-i][j].
    "and column-wise" means cell[i][j] == cell[i][N-1-j].
    "and cell == its transpose" means cell[i][j] == cell[j][i].
    This is the D4 symmetry group (square symmetry).
    """
    N = len(cell)
    # Convert to numpy array for manipulation
    c = np.array(cell, dtype=float)
    
    # Average over all 8 symmetry operations to ensure symmetry
    # Operations:
    # 1. Identity
    # 2. Rotate 90
    # 3. Rotate 180
    # 4. Rotate 270
    # 5. Reflect X (left-right)
    # 6. Reflect Y (up-down)
    # 7. Reflect Diag (transpose)
    # 8. Reflect Anti-Diag
    
    sym_ops = [
        lambda a: a,
        lambda a: np.rot90(a, 1),
        lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a, 3),
        lambda a: np.fliplr(a),
        lambda a: np.flipud(a),
        lambda a: a.T,
        lambda a: np.fliplr(np.rot90(a, 1)) # Anti-diagonal reflection
    ]
    
    avg = np.zeros_like(c)
    for op in sym_ops:
        avg += op(c)
    avg /= 8.0
    
    # Threshold to binary
    final = (avg > 0.5).astype(int)
    
    # Verify symmetry strictly
    # If thresholding broke symmetry due to .5 values, force it by taking the mode or just using the top-left quadrant and mirroring.
    # Better approach: Define top-left quadrant and mirror.
    q = c[:N//2, :N//2]
    # Since N=32, N//2 = 16.
    # Top-left is 16x16.
    # Mirror to top-right, bottom-left, bottom-right.
    
    # To strictly satisfy cell == cell.T, we need symmetry across the main diagonal.
    # This implies the cell is determined by the upper triangle (including diagonal) or just the first quadrant if we are careful.
    # Actually, D4 symmetry on an even grid (32x32) means the center is between pixels.
    # The fundamental domain is 16x16.
    
    # Let's reconstruct from the average to be safe, but force binary consistency.
    # If we just return the thresholded average, it should be symmetric if the input was close.
    # But to be absolutely safe, let's enforce by copying the top-left quadrant to others.
    
    # However, "cell == its transpose" is a strong constraint.
    # Let's use the averaged result, threshold it, and then explicitly enforce the symmetries by averaging again and thresholding.
    
    return final.tolist()

def _compute_gap_approx(cell, num_g=5, num_k=5):
    """
    Fast approximate band structure calculation using FFT-based dielectric averaging.
    This is an approximation to the exact Fourier coefficient method, but fast enough for search.
    """
    N = len(cell)
    eps = np.array(cell, dtype=float)
    eps = np.where(eps == 1, 12.0, 1.0) # 1 is dielectric (12), 0 is air (1)
    
    # Inverse epsilon for TM mode
    inv_eps = 1.0 / eps
    inv_eps_avg = np.mean(inv_eps)
    if inv_eps_avg == 0:
        return -1.0
    inv_eps_norm = inv_eps / inv_eps_avg
    
    # Define G-vectors
    # We use a square grid of G-vectors from -num_g to num_g
    g_vals = np.arange(-num_g, num_g + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    G = np.stack([Gx.ravel(), Gy.ravel()], axis=1).astype(float)
    NG = len(G)
    
    # Compute Fourier coefficients of inv_eps_norm using FFT
    # The pixel map is a 32x32 grid representing one unit cell.
    # We need the Fourier series coefficients c_G.
    # c_G = (1/N^2) sum_{i,j} eps_inv(i,j) exp(-i G . r_ij)
    # This is exactly the DFT of the 2D array, scaled.
    
    # Perform 2D FFT
    fft_coeffs = np.fft.fft2(inv_eps_norm)
    # The FFT output is ordered such that index [0,0] is G=0, [1,0] is G=(1,0), etc.
    # We need to map our G-vectors to these indices.
    # G = (gx, gy) where gx, gy are integers.
    # The FFT coefficient for G=(gx, gy) is at index [gx mod N, gy mod N] if we consider the standard DFT definition.
    # However, np.fft.fft2 gives frequencies from 0 to N-1.
    # Negative frequencies are wrapped around.
    # Let's extract the coefficients for our specific G-vectors.
    
    coeffs = np.zeros(NG, dtype=complex)
    for idx, (gx, gy) in enumerate(G):
        # Map gx, gy to 0..N-1
        ix = int(gx) % N
        iy = int(gy) % N
        coeffs[idx] = fft_coeffs[iy, ix] / (N * N) # Normalization
        
    # Build the Hamiltonian matrix for each k-point
    # H_{G, G'} = (G + k) . (G' + k) * c_{G - G'}  (for TM, with appropriate factors)
    # Actually, for TM: H_{G, G'} = (G+k) . (G'+k) * <1/eps>_{G-G'}
    # The eigenvalues are proportional to omega^2.
    
    # Irreducible Brillouin Zone k-points
    # Square lattice, BZ boundaries at pi, -pi.
    # Symmetry points: Gamma (0,0), M (pi,0), X (pi/2, pi/2) -- wait, X is (pi,0) in some conventions?
    # Standard: Gamma (0,0), M (pi,0), X (pi, pi/2)? No.
    # For square lattice:
    # Gamma: (0,0)
    # M: (pi, 0) (center of edge)
    # X: (pi/2, pi/2) (corner) -- often called K or X depending on convention.
    # Let's use a few points along the path Gamma -> M -> X -> Gamma.
    
    k_points = []
    # Gamma to M
    for i in range(num_k):
        t = i / (num_k - 1)
        k_points.append((t * np.pi, 0))
    # M to X
    for i in range(1, num_k):
        t = i / (num_k - 1)
        k_points.append((np.pi - t * np.pi/2, t * np.pi/2))
    # X to Gamma
    for i in range(1, num_k):
        t = i / (num_k - 1)
        k_points.append((t * np.pi/2, (1-t) * np.pi/2))
        
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        # Compute (G+k) dot (G'+k)
        # G is NG x 2
        # Gp is NG x 2
        # We need the outer product of (G+k) and (G'+k) for each pair?
        # No, H_{ij} = (G_i + k) . (G_j + k) * c_{G_i - G_j}
        
        Gk = G + k
        # Dot products: Gk_i . Gk_j
        # This is an NG x NG matrix.
        # dot_prods = Gk @ Gk.T  -- This gives G_i . G_j, but we need G_i . G_j where both are shifted by k?
        # Yes, (G_i + k) . (G_j + k).
        dot_prods = Gk @ Gk.T
        
        # Now we need to multiply each element (i,j) by c_{G_i - G_j}
        # We need to find the index in 'coeffs' corresponding to G_i - G_j.
        # Precompute a map from G-vector tuple to index
        g_to_idx = {}
        for idx, (gx, gy) in enumerate(G):
            g_to_idx[(int(gx), int(gy))] = idx
            
        c_mat = np.zeros((NG, NG), dtype=complex)
        for i in range(NG):
            for j in range(NG):
                dx = int(G[i, 0] - G[j, 0])
                dy = int(G[i, 1] - G[j, 1])
                if (dx, dy) in g_to_idx:
                    c_mat[i, j] = coeffs[g_to_idx[(dx, dy)]]
                else:
                    c_mat[i, j] = 0.0
                    
        H = dot_prods * c_mat
        # H is Hermitian
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            # Take the first few bands. We want bands 4 and 5 (1-indexed).
            # Indices 3 and 4 (0-indexed).
            if len(eigvals) > 4:
                w4 = np.sqrt(eigvals[3])
                w5 = np.sqrt(eigvals[4])
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        except np.linalg.LinAlgError:
            continue
            
    if min_omega_5 == float('inf'):
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return -1.0
    return gap / midgap

def _generate_candidate(params):
    """
    Generate a symmetric cell based on parameters.
    params: [r_center, arm_width, arm_length, arm_thickness]
    """
    N = 32
    cell = np.zeros((N, N), dtype=int)
    cx, cy = N/2.0, N/2.0
    
    r_c = params[0]
    a_w = params[1]
    a_l = params[2]
    a_t = params[3]
    
    # Central disk
    for i in range(N):
        for j in range(N):
            dx = i - cx
            dy = j - cy
            if dx*dx + dy*dy < r_c*r_c:
                cell[i, j] = 1
                
    # Arms along x and y axes
    # Arm 1: Right (+x)
    # Extends from center to cx + a_l
    # Width a_w, thickness a_t? 
    # Let's define arms as rectangles extending from the center.
    # Right arm: x in [cx, cx+a_l], y in [cy-a_w/2, cy+a_w/2]
    # But we need to handle pixel indices.
    
    for axis in ['x', 'y']:
        for direction in [1, -1]:
            # Determine start and end
            if axis == 'x':
                if direction == 1:
                    start = int(cx)
                    end = int(cx + a_l)
                else:
                    start = int(cx - a_l)
                    end = int(cx)
            else:
                if direction == 1:
                    start = int(cy)
                    end = int(cy + a_l)
                else:
                    start = int(cy - a_l)
                    end = int(cy)
            
            # Width
            half_w = a_w / 2.0
            
            if axis == 'x':
                # Horizontal arm
                for i in range(start, end + 1):
                    if 0 <= i < N:
                        for j in range(int(cy - half_w), int(cy + half_w) + 1):
                            if 0 <= j < N:
                                cell[i, j] = 1
            else:
                # Vertical arm
                for j in range(start, end + 1):
                    if 0 <= j < N:
                        for i in range(int(cy - half_w), int(cy + half_w) + 1):
                            if 0 <= i < N:
                                cell[i, j] = 1
                                
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    N = problem["pixels"]
    eps_high = problem["eps_high"]
    
    # Initial parameters
    # r_center: 2 to 10
    # arm_width: 2 to 10
    # arm_length: 5 to 15
    # arm_thickness: not used in current gen, kept for compatibility
    
    best_params = [5.0, 4.0, 8.0, 0.0]
    best_score = -1.0
    best_cell = _generate_candidate(best_params)
    
    # Local search
    # Define a small step size
    steps = [0.5, 0.2, 0.1]
    
    current_params = best_params[:]
    current_score = _compute_gap_approx(_generate_candidate(current_params), num_g=4, num_k=4)
    
    for step in steps:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try perturbing each parameter
            for i in range(len(current_params) - 1): # Ignore thickness for now
                for delta in [-step, step]:
                    new_params = current_params[:]
                    new_params[i] += delta
                    # Bounds
                    if new_params[i] < 1.0 or new_params[i] > 15.0:
                        continue
                    
                    new_cell = _generate_candidate(new_params)
                    new_score = _compute_gap_approx(new_cell, num_g=4, num_k=4)
                    
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params[:]
                            best_cell = new_cell
                            
            # If no improvement, break
            if not improved:
                break
                
    # Final refinement with higher accuracy if time permits
    if time.time() < deadline - 10:
        # Try a few random perturbations around the best found
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            rand_params = [p + np.random.uniform(-1, 1) for p in best_params]
            rand_params = [max(1.0, min(15.0, p)) for p in rand_params]
            rand_cell = _generate_candidate(rand_params)
            rand_score = _compute_gap_approx(rand_cell, num_g=5, num_k=5)
            if rand_score > best_score:
                best_score = rand_score
                best_params = rand_params
                best_cell = rand_cell
                
    # Ensure the final cell is strictly symmetric
    final_cell = _make_symmetric(best_cell)
    
    return final_cell

