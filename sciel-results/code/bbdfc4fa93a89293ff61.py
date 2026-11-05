import numpy as np
import time
import math

def _build_symmetric_cell(size, quadrant):
    """
    quadrant: 16x16 boolean array representing top-left 16x16.
    Expands to 32x32 using C4v symmetry.
    """
    N = size
    half = N // 2
    cell = np.zeros((N, N), dtype=int)
    # Copy quadrant to top-left
    cell[:half, :half] = quadrant[:half, :half]
    
    # Reflect to top-right (transpose within quadrant? No, global transpose)
    # cell[i][j] = cell[j][i]
    # cell[i][31-j] = cell[i][j] (col flip)
    # cell[31-i][j] = cell[i][j] (row flip)
    
    # 1. Fill top-right from top-left via col flip? No, C4v implies:
    # Top-Left (TL) -> Top-Right (TR) via x->N-1-x? No.
    # Let's just use the rules:
    # cell[i][j] is defined for i<16, j<16.
    # cell[i][N-1-j] = cell[i][j]
    # cell[N-1-i][j] = cell[i][j]
    # cell[i][j] = cell[j][i]
    
    # We have TL.
    # TR[i][j] = TL[i][N-1-j]? No.
    # If we reflect TL across y-axis (x=15.5), we get TR.
    # TL[i][j] corresponds to x<16. TR[i][k] with x>15.
    # x_TR = N-1-x_TL.
    # So cell[i][N-1-j] = cell[i][j].
    for i in range(half):
        for j in range(half):
            cell[i, N-1-j] = cell[i, j]
            
    # Now we have Top Row (0..15).
    # Reflect across x-axis (y=15.5) to get Bottom Row.
    # cell[N-1-i][j] = cell[i][j].
    for i in range(half):
        for j in range(N):
            cell[N-1-i, j] = cell[i, j]
            
    # Check Transpose Symmetry: cell[i][j] == cell[j][i]
    # The above construction ensures row/col flips.
    # Does it ensure transpose?
    # cell[i][j] (i<16, j<16) = TL[i][j]
    # cell[j][i] (j<16, i<16) = TL[j][i]
    # We need TL[i][j] == TL[j][i].
    # So the input quadrant MUST be symmetric about its main diagonal.
    
    return cell.tolist()

def _pwe_gap(cell, eps_low=1.0, eps_high=12.0, lower_band=4, upper_band=5, num_pw=17, num_k=30):
    """
    Calculate the gap ratio between lower_band and upper_band TM modes.
    num_pw: number of G vectors in one direction (odd).
    """
    N = len(cell)
    # Discretize epsilon
    eps = np.array(cell, dtype=float)
    eps[eps == 1] = eps_high
    eps[eps == 0] = eps_low
    
    # Fourier coefficients of epsilon
    # G vectors
    G_max = num_pw // 2
    G_vals = np.arange(-G_max, G_max + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    num_G = len(Gx)
    
    # Calculate eps_G
    # eps_G = (1/N^2) * sum_{i,j} eps[i,j] * exp(-i * (Gx[i] + Gy[j]) * 2pi * (i/N, j/N))
    # Vectorized:
    # Coordinates in [0, 1)
    x = np.arange(N) / N
    y = np.arange(N) / N
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    eps_flat = eps.flatten()
    
    # Phase matrix: num_G x N*N
    # exp(-2j * pi * (Gx * xx + Gy * yy))
    # Gx is num_G, xx is N*N
    phase = np.exp(-2j * np.pi * (Gx[:, None] * xx_flat[None, :] + Gy[:, None] * yy_flat[None, :]))
    eps_G = np.sum(phase * eps_flat[None, :], axis=1) / (N * N)
    
    # Map G indices to array indices for convolution
    # G indices: -G_max ... G_max
    # Index in array: G + G_max
    def g_idx(gx, gy):
        return (gx + G_max) * (2 * G_max + 1) + (gy + G_max)
        
    # Precompute epsilon differences? No, we need eps_G - eps_G' for each pair.
    # Matrix A for PWE:
    # A[k, G, G'] = eps_{G-G'} * |k + G'|^2 ? 
    # Wait, the equation is:
    # sum_{G'} eps_{G-G'} * |k+G'|^2 * a_{G'} = omega^2 * a_G
    # This is not symmetric.
    # Symmetric form:
    # Let b_G = a_G.
    # M_{G, G'} = sqrt(eps_{G-G'} eps_{G'-G}) ? No.
    # Standard symmetric TM PWE:
    # M_{G, G'} = ( (k+G) . (k+G') ) / eps_{G-G'} ? 
    # Let's check literature.
    # Joannopoulos: 
    # TE: M_{G,G'} = (k+G) . (k+G') / eps_{G-G'}
    # TM: M_{G,G'} = (k+G) . (k+G') * eps_{G-G'} ? 
    # Actually, for TM, the operator is -1/eps * grad x grad x.
    # In Fourier space:
    # sum_{G'} ( (k+G) . (k+G') / eps_{G-G'} ) a_{G'} ? No.
    # Let's use the non-symmetric form and use eigvals. It's safer.
    
    # K points in irreducible BZ
    # Square lattice BZ: -pi/a to pi/a.
    # Irreducible: 0 <= kx <= ky <= pi/a? Or 0 <= kx <= pi/a, 0 <= ky <= kx?
    # Usually: 0 <= kx <= pi/a, 0 <= ky <= pi/a, kx <= ky.
    # Let's sample points in the triangle (0,0) - (pi,0) - (pi/2, pi/2)?
    # No, the 1/8th BZ is the triangle with vertices (0,0), (pi,0), (pi/2, pi/2).
    # Wait, the symmetry group C4v reduces the BZ to 1/8.
    # Vertices: Gamma (0,0), X (pi, 0), M (pi/2, pi/2).
    # We will sample a grid in this triangle.
    
    k_points = []
    # Simple grid in the triangle
    n_k = 5
    for i in range(n_k + 1):
        for j in range(n_k + 1 - i):
            # Map (i,j) to (kx, ky)
            # kx = i/n_k * pi
            # ky = j/n_k * pi
            # Constraint: kx <= ky? 
            # Let's just use a dense grid in the square [0, pi]x[0, pi] and filter kx <= ky?
            pass
            
    # Better: Linear interpolation along the boundary and interior.
    # Let's just use a fixed set of high-symmetry points and some interior points.
    pts = [
        (0, 0), # Gamma
        (math.pi, 0), # X
        (math.pi/2, math.pi/2), # M
        (math.pi/4, math.pi/4), # Midpoint Gamma-M
        (3*math.pi/4, math.pi/4), # Midpoint X-M? No, X is (pi,0). M is (pi/2, pi/2).
        # Line X-M: kx from pi to pi/2, ky from 0 to pi/2.
        (3*math.pi/4, math.pi/4),
        (0, math.pi/2), # This is outside kx<=ky? 0 <= pi/2. Yes.
        (math.pi/2, 0), # Outside kx<=ky? pi/2 > 0. No.
    ]
    # Let's generate a proper grid in the 1/8 BZ.
    # 0 <= kx <= pi, 0 <= ky <= pi, kx <= ky.
    # Actually, the standard 1/8 BZ for square lattice is:
    # 0 <= kx <= pi/2, 0 <= ky <= pi/2, kx <= ky? No.
    # The full BZ is -pi to pi.
    # 1/8 is the triangle (0,0), (pi,0), (pi/2, pi/2) is 1/4?
    # Area of BZ is 4 pi^2. Area of triangle is 1/2 * pi * pi/2 = pi^2/4.
    # Ratio is 1/16. So it's 1/16.
    # C4v has 8 elements. So 1/8 BZ should have area pi^2/2.
    # The region 0 <= kx <= pi, 0 <= ky <= pi, kx <= ky is a triangle with area pi^2/2.
    # Yes, this is the 1/8 BZ.
    
    k_grid = []
    n_k = 6
    for i in range(n_k + 1):
        kx = i * math.pi / n_k
        for j in range(n_k + 1 - i):
            ky = (n_k - i + j) * math.pi / n_k
            # Check if kx <= ky
            if kx <= ky + 1e-9:
                k_grid.append((kx, ky))
                
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for kx, ky in k_grid:
        k = np.array([kx, ky])
        # Build Matrix A
        # A[G, G'] = eps_{G-G'} * |k+G'|^2
        # G vectors are in units of 2pi/a.
        # |k+G'|^2 = (kx+Gx')^2 + (ky+Gy')^2
        
        k_plus_G = np.sqrt((kx + Gx)**2 + (ky + Gy)**2)
        
        # We need eps_{G-G'} for all pairs.
        # G-G' corresponds to difference in indices.
        # This is a circulant-like structure but on a 2D grid.
        
        # Efficient way:
        # For each G, we need the vector of eps_{G-G'} for all G'.
        # This is a convolution.
        
        # Let's just loop over G (outer) and G' (inner) for simplicity, 
        # but 289x289 is small.
        A = np.zeros((num_G, num_G), dtype=complex)
        
        for i in range(num_G):
            gx_i = Gx[i]
            gy_i = Gy[i]
            for j in range(num_G):
                gx_j = Gx[j]
                gy_j = Gy[j]
                dgx = gx_i - gx_j
                dgy = gy_i - gy_j
                # Get eps_{dgx, dgy}
                idx_dgx = int(dgx + G_max)
                idx_dgy = int(dgy + G_max)
                eps_diff = eps_G[idx_dgx * (2 * G_max + 1) + idx_dgy]
                
                # |k + G_j|^2
                k_sq = (kx + gx_j)**2 + (ky + gy_j)**2
                
                A[i, j] = eps_diff * k_sq
                
        # Solve Eigenvalues
        # A is not symmetric.
        w = np.linalg.eigvals(A)
        w = np.sort(np.real(w))
        
        # Bands are indexed 1..N. We want 4 and 5.
        # w[0] is band 1.
        # w[3] is band 4.
        # w[4] is band 5.
        
        if len(w) >= 5:
            omega_4 = w[3]
            omega_5 = w[4]
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
            
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Candidate Generators
    def gen_rod(radius):
        quad = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                # Center of 32x32 is 15.5, 15.5
                # Center of 16x16 quad is 7.5, 7.5?
                # The quad represents pixels 0..15.
                # Pixel (i,j) in quad maps to (i,j) in cell.
                # Distance from cell center (15.5, 15.5)
                dx = i - 15.5
                dy = j - 15.5
                if dx*dx + dy*dy < radius*radius:
                    quad[i, j] = 1
        # Symmetrize quad (transpose)
        quad = (quad + quad.T) // 2
        return quad

    def gen_cross(width):
        quad = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                # Cross centered at 15.5, 15.5
                # Arms along x and y
                # |i - 15.5| < width/2 AND |j - 15.5| < width/2? No, that's a square.
                # Cross: (|i - 15.5| < w/2) OR (|j - 15.5| < w/2)
                if abs(i - 15.5) < width/2.0 or abs(j - 15.5) < width/2.0:
                    quad[i, j] = 1
        quad = (quad + quad.T) // 2
        return quad

    def gen_air_cross(width):
        # Dielectric background with air cross
        quad = np.ones((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                if abs(i - 15.5) < width/2.0 or abs(j - 15.5) < width/2.0:
                    quad[i, j] = 0
        quad = (quad + quad.T) // 2
        return quad

    def gen_square(size_s):
        quad = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in range(16):
                if abs(i - 15.5) < size_s/2.0 and abs(j - 15.5) < size_s/2.0:
                    quad[i, j] = 1
        quad = (quad + quad.T) // 2
        return quad

    # Search Parameters
    candidates = []
    
    # Rods
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(gen_rod(r))
        
    # Crosses (Dielectric in Air)
    for w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        candidates.append(gen_cross(w))
        
    # Air Crosses (Air in Dielectric)
    for w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        candidates.append(gen_air_cross(w))
        
    # Squares
    for s in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(gen_square(s))
        
    best_score = -1.0
    best_cell = None
    best_quad = candidates[0]
    
    # Evaluate candidates
    for quad in candidates:
        if time.time() > deadline:
            break
        cell = _build_symmetric_cell(size, quad)
        score = _pwe_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            best_quad = quad
            
    # If time remains, do a local search on the best quad
    if time.time() < deadline - 30.0 and best_quad is not None:
        current_quad = best_quad.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try flipping pixels in the fundamental domain (i<=j, i<16, j<16)
            for i in range(16):
                for j in range(i, 16):
                    # Flip pixel
                    old_val = current_quad[i, j]
                    current_quad[i, j] = 1 - old_val
                    
                    # Symmetrize
                    sym_quad = (current_quad + current_quad.T) // 2
                    cell = _build_symmetric_cell(size, sym_quad)
                    score = _pwe_gap(cell)
                    
                    if score > current_score:
                        current_score = score
                        best_score = score
                        best_cell = cell
                        best_quad = sym_quad.copy()
                        improved = True
                    else:
                        # Revert
                        current_quad[i, j] = old_val
                        
    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell
