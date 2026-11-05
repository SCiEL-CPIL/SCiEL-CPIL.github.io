import numpy as np
import time
import scipy.linalg

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0
    
    size = problem["pixels"]
    
    # Precompute k-points
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx+ky <= 1 (in units of 2pi/a)
    # Actually, standard is: 0 <= kx <= 0.5, 0 <= ky <= 0.5, kx <= ky is not the full IRBZ.
    # IRBZ is the triangle with vertices (0,0), (0.5,0), (0.5,0.5)? No.
    # High symmetry points: Gamma (0,0), X (0.5,0), M (0.5,0.5).
    # The IRBZ is the region bounded by kx=0, ky=kx, and kx+ky=0.5?
    # Let's use a dense grid over the triangle (0,0)-(0.5,0)-(0.5,0.5) or similar.
    # Standard set: 
    # kx from 0 to 0.5, ky from 0 to kx? No.
    # The BZ is [-0.5, 0.5]x[-0.5, 0.5].
    # Symmetry group C4v. IRBZ is 1/8 of the square.
    # Vertices: (0,0), (0.5,0), (0.5,0.5).
    # Region: 0 <= kx <= 0.5, 0 <= ky <= kx.
    # Let's generate 30 points in this triangle.
    
    k_points = []
    # Add high symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    k_points.append((0.25, 0.0))
    k_points.append((0.25, 0.25))
    k_points.append((0.0, 0.0)) # Gamma
    # Add random points in the triangle
    for _ in range(25):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        # Map (u,v) in unit triangle to (kx,ky) in (0,0)-(0.5,0)-(0.5,0.5)
        # kx = 0.5 * u
        # ky = 0.5 * v * (1 - u) ? No.
        # Affine transform of (0,0), (1,0), (0,1) to (0,0), (0.5,0), (0.5,0.5)
        # P = (0,0) + u*(0.5,0) + v*(0,0.5) ?? No, that's a square.
        # Triangle vertices: A(0,0), B(0.5,0), C(0.5,0.5).
        # Point = A + u*(B-A) + v*(C-B) = (0.5u, 0.5v)
        # Constraints: u>=0, v>=0, u+v<=1.
        kx = 0.5 * u
        ky = 0.5 * v
        # Wait, if u=1, v=0 -> (0.5, 0). If u=0, v=1 -> (0, 0.5). This is wrong.
        # B-A = (0.5, 0). C-B = (0, 0.5).
        # So (0.5u, 0.5v).
        # If u=1, v=0 -> (0.5, 0). Correct.
        # If u=0, v=1 -> (0, 0.5). Correct.
        # If u=0.5, v=0.5 -> (0.25, 0.25). Correct.
        # The region u+v<=1 maps to kx/0.5 + ky/0.5 <= 1 => kx+ky <= 0.5.
        # This is the triangle (0,0)-(0.5,0)-(0,0.5).
        # But the IRBZ is (0,0)-(0.5,0)-(0.5,0.5).
        # The line ky=kx is the symmetry axis.
        # So we need 0 <= ky <= kx.
        # My mapping gives a triangle with hypotenuse kx+ky=0.5.
        # The IRBZ triangle has hypotenuse ky = 0.5 - kx? No.
        # The BZ is square [-0.5, 0.5]^2.
        # C4 symmetry axes: kx=0, ky=0, kx=ky, kx=-ky.
        # IRBZ is 1/8.
        # Vertices: (0,0), (0.5,0), (0.5,0.5).
        # Boundaries: kx=0 (y-axis), ky=kx (diagonal), ky=0.5 (top edge? No).
        # The boundary of the BZ is kx=0.5 and ky=0.5.
        # So the region is 0 <= kx <= 0.5, 0 <= ky <= kx.
        # Let's generate points in 0 <= ky <= kx <= 0.5.
        for _ in range(25):
            kx = np.random.uniform(0, 0.5)
            ky = np.random.uniform(0, kx)
            k_points.append((kx, ky))
            
    k_points = np.array(k_points)
    
    # Fourier Coefficients
    # G vectors
    M_g = 8 # 17x17 = 289
    G_x = np.arange(-M_g, M_g + 1)
    G_y = np.arange(-M_g, M_g + 1)
    G_x, G_y = np.meshgrid(G_x, G_y, indexing='ij')
    G_flat = np.stack([G_x.ravel(), G_y.ravel()], axis=1).astype(float)
    
    # Pixel centers
    i_idx, j_idx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_pix = (i_idx + 0.5) / size - 0.5
    y_pix = (j_idx + 0.5) / size - 0.5
    pix_flat = np.stack([x_pix.ravel(), y_pix.ravel()], axis=1)
    
    # Precompute exp(-i G . r)
    # G_flat: (N_G, 2), pix_flat: (N_pix, 2)
    # Dot product: (N_G, N_pix)
    dots = G_flat @ pix_flat.T
    exp_terms = np.exp(-1j * dots)
    
    def get_eps_fourier(cell):
        # cell: 32x32, 0 or 1
        # eps = 1 + (12-1) * cell
        eps_pix = 1.0 + 11.0 * np.array(cell).flatten()
        # eps_G = 1/N_pix * sum(eps_pix * exp_terms)
        eps_G = np.sum(eps_pix[None, :] * exp_terms, axis=1) / (size*size)
        return eps_G

    def evaluate_cell(cell):
        eps_G = get_eps_fourier(cell)
        # Map G index to G vector
        # We need eps_{G - G'}
        # Let's create a dictionary or 2D array for eps_G
        # G_x, G_y are integers from -M_g to M_g
        # Index mapping: idx = (gx + M_g) * (2*M_g + 1) + (gy + M_g)
        N_g = 2 * M_g + 1
        eps_G_2d = np.zeros((N_g, N_g), dtype=complex)
        for i in range(N_g):
            for j in range(N_g):
                eps_G_2d[i, j] = eps_G[i * N_g + j]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build Matrix M
            # M[G, G'] = eps_{G-G'} * (k+G')^2
            # (k+G')^2 is (kx+Gx')^2 + (ky+Gy')^2
            k_vec = np.array([kx, ky])
            G_plus_k = G_flat + k_vec
            k2 = np.sum(G_plus_k**2, axis=1)
            
            # M is N_G x N_G
            # M[i, j] = eps_G_2d[gx_i - (-M_g), gy_i - (-M_g)] ... wait
            # G vector for index i: G_x[i//N_g], G_y[i%N_g]
            # We need eps_{G_i - G_j}
            # G_i - G_j has components (Gx_i - Gx_j, Gy_i - Gy_j)
            # Index in eps_G_2d: (Gx_i - Gx_j + M_g, Gy_i - Gy_j + M_g)
            
            M_mat = np.zeros((N_g**2, N_g**2), dtype=complex)
            # This loop is slow. Vectorize?
            # Let's precompute the index differences
            # G_flat is (N_G, 2)
            # G_diff = G_flat[:, None, :] - G_flat[None, :, :]
            # G_diff: (N_G, N_G, 2)
            # idx_x = G_diff[..., 0] + M_g
            # idx_y = G_diff[..., 1] + M_g
            # eps_diff = eps_G_2d[idx_x, idx_y]
            
            G_diff = G_flat[:, None, :] - G_flat[None, :, :]
            idx_x = G_diff[..., 0].astype(int) + M_g
            idx_y = G_diff[..., 1].astype(int) + M_g
            eps_diff = eps_G_2d[idx_x, idx_y]
            
            # M[i, j] = eps_diff[i, j] * k2[j]
            M_mat = eps_diff * k2[None, :]
            
            # Solve eigvals
            # M is not symmetric.
            eigvals = np.linalg.eigvals(M_mat)
            # Sort by real part
            eigvals_real = np.sort(np.real(eigvals))
            
            # We need bands 4 and 5 (indices 3 and 4)
            if len(eigvals_real) >= 5:
                w4 = eigvals_real[3]
                w5 = eigvals_real[4]
                if w5 < min_omega_5: min_omega_5 = w5
                if w4 > max_omega_4: max_omega_4 = w4
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0: return 0
        return gap / midgap

    # Search
    best_score = -1.0
    best_cell = None
    
    # Initial guesses: Air holes
    # Radius in pixels: 4 to 12
    # Shape: Super-ellipse |x|^n + |y|^n = R^n
    
    def make_cell(radius, n, is_air_hole=True):
        cell = np.zeros((size, size), dtype=int)
        if is_air_hole:
            cell[:] = 1 # Dielectric
        else:
            cell[:] = 0 # Air
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Super-ellipse
                val = abs(x)**n + abs(y)**n
                if is_air_hole:
                    if val < radius**n:
                        cell[i, j] = 0 # Air
                else:
                    if val < radius**n:
                        cell[i, j] = 1 # Dielectric
        return cell.tolist()

    # Random Search
    np.random.seed(42)
    
    num_iters = 0
    while time.time() < deadline:
        # Sample parameters
        # Air hole
        r = np.random.uniform(0.15, 0.45) # In units of lattice constant
        n = np.random.uniform(1.0, 6.0)
        
        cell = make_cell(r, n, True)
        
        # Quick check: ensure symmetry (it is by construction)
        # Ensure not empty/full
        if np.sum(cell) == 0 or np.sum(cell) == size*size:
            continue
            
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
        num_iters += 1
        
        # Local search around best?
        # If we have time, perturb the best cell
        if num_iters % 10 == 0 and best_cell is not None:
             # Perturb boundary pixels
             pass

    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell
