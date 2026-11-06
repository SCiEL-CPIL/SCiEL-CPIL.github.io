import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap
    between bands 4 and 5.
    
    Strategy:
    1. Start with a 2x2 array of dielectric rods (known good starting point).
    2. Use a local search in pixel space, respecting D4 symmetry.
    3. Use a fast plane-wave method to evaluate the gap.
    4. Use a deadline guard to ensure we return within the time budget.
    """
    size = problem["pixels"]
    if size != 32:
        size = 32
        
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s budget
    
    # D4 symmetry: cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j]
    # and cell[i][j] == cell[j][i]
    # So we only need to specify the top-left 16x16 quadrant.
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    # Initial design: 2x2 array of rods
    # Rod centers at (8,8), (8,24), (24,8), (24,24) in 32x32 cell
    # In the 16x16 quadrant, this corresponds to a single rod at (8,8)
    # with radius ~0.14 * 32 = 4.48 pixels
    
    radius = 0.14
    r_px = radius * size
    
    q_init = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            x = (j + 0.5) / 16.0
            y = (i + 0.5) / 16.0
            # Distance to rod center at (0.5, 0.5) in quadrant coords
            # which is (8.5, 8.5) in full cell coords
            dx = x - 0.5
            dy = y - 0.5
            if dx*dx + dy*dy < (r_px/16.0)**2:
                q_init[i, j] = 1
    
    best_q = q_init.copy()
    best_cell = quadrant_to_full(best_q)
    best_gap = -np.inf
    
    # Fast gap evaluation
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a plane-wave expansion with 289 plane waves (npw=8).
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        inv_eps = 1.0 / eps
        
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        # Key k-points in the irreducible BZ
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build the matrix M[G, G'] = (k+G)^2 * (1/eps)_{G-G'}
            # We need the inverse dielectric Fourier coefficients
            # (1/eps)_{G-G'} for all pairs G, G'
            
            # For efficiency, precompute the difference indices
            M = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    diff = G[i] - G[j]
                    # Find the index of diff in G
                    # diff should be in G since both are in the same grid
                    # But we need to check if it's within the cutoff
                    if abs(diff[0]) <= 2*np.pi*npw and abs(diff[1]) <= 2*np.pi*npw:
                        mx = int(round(diff[0] / (2*np.pi)))
                        my = int(round(diff[1] / (2*np.pi)))
                        # Find index in G
                        idx = None
                        for k in range(nG):
                            if G[k][0] == diff[0] and G[k][1] == diff[1]:
                                idx = k
                                break
                        if idx is not None:
                            M[i, j] = k_plus_G_sq[i] * inv_eps_coeffs[idx]
            
            # Solve eigenvalue problem
            try:
                eigenvalues = np.linalg.eigvalsh(M)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = np.sort(eigenvalues)
                
                # Get bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -np.inf
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap == 0:
            return -np.inf
        
        return 2 * gap / midgap
    
    # Evaluate initial design
    best_gap = compute_gap(best_cell)
    
    # Local search: flip pixels in the quadrant
    # Respect D4 symmetry
    
    def get_symmetry_orbit(i, j):
        """Get all symmetric positions for (i, j) in the 16x16 quadrant."""
        # In the 16x16 quadrant, the symmetry operations are:
        # - Identity: (i, j)
        # - Reflection about vertical center: (i, 15-j) -- but this is in the full cell
        # Actually, within the quadrant, we just have the single pixel (i, j)
        # because the quadrant is the fundamental domain.
        # But we need to be careful: the quadrant is 16x16, and the full cell is 32x32.
        # The quadrant covers rows 0-15 and cols 0-15.
        # The symmetry operations map within the full cell, but within the quadrant,
        # each pixel is unique.
        return [(i, j)]
    
    # Local search
    search_radius = 3  # Only consider pixels within this distance of the rod center
    rod_center_q = (8, 8)  # In quadrant coords
    
    # Get pixels in the search region
    search_pixels = []
    for i in range(16):
        for j in range(16):
            dx = i - rod_center_q[0]
            dy = j - rod_center_q[1]
            if dx*dx + dy*dy <= search_radius*search_radius:
                search_pixels.append((i, j))
    
    # Also add some pixels further out for more flexibility
    for i in range(16):
        for j in range(16):
            dx = i - rod_center_q[0]
            dy = j - rod_center_q[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if search_radius < dist <= search_radius + 2:
                search_pixels.append((i, j))
    
    np.random.seed(42)
    
    num_iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline and num_iterations < max_iterations:
        # Pick a random pixel to flip
        idx = np.random.randint(len(search_pixels))
        i, j = search_pixels[idx]
        
        # Flip the pixel
        q_new = best_q.copy()
        q_new[i, j] = 1 - q_new[i, j]
        
        # Evaluate
        cell_new = quadrant_to_full(q_new)
        gap_new = compute_gap(cell_new)
        
        if gap_new > best_gap:
            best_q = q_new
            best_cell = cell_new
            best_gap = gap_new
        
        num_iterations += 1
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Convert to list of lists
    result = [[int(x) for x in row] for row in best_cell]
    
    return result
