"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Search over 2x2 rod configurations (position, radius) using a fast
    plane-wave proxy to maximize the TM band-4/5 gap ratio.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 200s limit
    
    def enforce_d4(cell):
        """Enforce D4 symmetry on a numpy array."""
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        cell = np.minimum(cell, cell[::-1, :])
        cell = np.minimum(cell, cell[:, ::-1])
        cell = np.minimum(cell, cell.T)
        return cell.astype(int)
    
    def make_cell(cx, cy, radius):
        """Create a 32x32 cell with 4 rods at D4-symmetric positions."""
        cell = np.zeros((size, size), dtype=int)
        r_sq = radius * radius
        # Only need to fill one quadrant and replicate
        # Quadrant: rows 0..15, cols 0..15 (top-left)
        # Rod center in this quadrant: (cx, cy) where cx, cy in [0, 16)
        for i in range(16):
            for j in range(16):
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy
                if dx*dx + dy*dy < r_sq:
                    cell[i][j] = 1
        # Replicate to all 4 quadrants with D4 symmetry
        cell[16:, :16] = cell[:16, :16][::-1, :]  # bottom-left: flip rows
        cell[:16, 16:] = cell[:16, :16][:, ::-1]   # top-right: flip cols
        cell[16:, 16:] = cell[:16, :16][::-1, ::-1] # bottom-right: flip both
        return cell
    
    def compute_eps_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon for the cell."""
        eps = np.where(cell == 1, eps_high, eps_low).astype(float)
        inv_eps = 1.0 / eps
        # Compute 2D FFT
        N = size
        coeffs = np.fft.fft2(inv_eps) / (N * N)
        return coeffs
    
    def plane_wave_gap(cell, n_pw=17, n_kpts=8):
        """
        Fast plane-wave approximation of the TM band gap ratio.
        Uses a truncated Fourier basis and a few k-points.
        """
        coeffs = compute_eps_fourier(cell)
        
        # Build reciprocal lattice vectors
        # For a square lattice with period 1, G = (mx, my) for integers m, n
        # We use a circular cutoff
        max_g = n_pw // 2
        g_vectors = []
        for mx in range(-max_g, max_g + 1):
            for my in range(-max_g, max_g + 1):
                g = (mx, my)
                g_sq = mx*mx + my*my
                if g_sq <= max_g * max_g:
                    g_vectors.append(g)
        
        n_g = len(g_vectors)
        
        # Map from (mx,my) to index
        g_index = {}
        for idx, g in enumerate(g_vectors):
            g_index[g] = idx
        
        # Build the TM eigenvalue matrix
        # For TM: (k - G)^2 * A_{G,G'} where A involves 1/epsilon
        # The eigenvalue equation: sum_G' [ (k-G)^2 delta_{G,G'} - sum_{G''} (k-G)·(k-G') * c_{G-G'} ] E_{G'} = 0
        # where c_n are Fourier coeffs of 1/epsilon
        
        # We'll use a simpler formulation:
        # For TM modes: [ (k-G)^2 / eps_G - sum_{G'} (k-G)·(k-G') / eps_{G-G'} ] E_{G'} = 0
        # Actually, the standard form is:
        # sum_{G'} [ (k-G)·(k-G') * c_{G-G'} ] E_{G'} = (k-G)^2 * E_G
        # where c_n = (1/N^2) sum_r exp(-i n·r) / eps(r)
        
        # Let's use: M_{G,G'} = (k-G)·(k-G') * c_{G-G'}
        # Eigenvalues give omega^2
        
        # k-points in irreducible BZ
        # Irreducible BZ of square lattice: 0 <= ky <= kx, kx^2 + ky^2 <= (pi)^2 (first BZ)
        # Actually for square lattice with a=1, BZ is [-pi, pi]^2
        # Irreducible: 0 <= ky <= kx, kx^2+ky^2 <= 2*pi^2... 
        # Let's use a few representative k-points
        
        k_points = []
        # Gamma
        k_points.append((0.0, 0.0))
        # M point (pi, 0)
        k_points.append((math.pi, 0.0))
        # X point (pi/2, pi/2) - actually for square lattice, X is at (pi, 0)
        # Let me reconsider: for square lattice a=1, reciprocal lattice vectors are 2*pi*(m,n)
        # BZ is [-pi, pi]^2
        # High symmetry: Gamma=(0,0), X=(pi,0), M=(pi,pi)
        k_points.append((math.pi, math.pi))
        # Some interior points
        k_points.append((math.pi/2, 0.0))
        k_points.append((math.pi/2, math.pi/2))
        k_points.append((0.75*math.pi, 0.0))
        k_points.append((0.75*math.pi, 0.5*math.pi))
        k_points.append((0.5*math.pi, 0.25*math.pi))
        
        # For each k-point, compute eigenvalues
        all_eigs = []
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build matrix
            M = np.zeros((n_g, n_g), dtype=complex)
            for ig, g in enumerate(g_vectors):
                kg = k - np.array(g) * 2 * math.pi  # k - G, where G = 2*pi*(mx,my)
                # Wait, for a=1 lattice, reciprocal lattice vectors are 2*pi*(m,n)
                # So G = 2*pi*(mx, my)
                for ig2, g2 in enumerate(g_vectors):
                    kg2 = k - np.array(g2) * 2 * math.pi
                    # G - G'
                    diff = np.array(g) - np.array(g2)
                    # Get Fourier coefficient c_{G-G'}
                    mx_diff, my_diff = diff[0], diff[1]
                    # coeffs[mx+size//2, my+size//2] gives c_{(mx,my)}
                    idx_m = int(mx_diff) % size
                    idx_n = int(my_diff) % size
                    c = coeffs[idx_m, idx_n]
                    # (k-G)·(k-G')
                    dot = kg[0]*kg2[0] + kg[1]*kg2[1]
                    M[ig, ig2] = dot * c
            
            # The eigenvalue equation is: M E = (k-G)^2 E
            # This is a generalized eigenvalue problem
            # Let's convert: M E = D E where D_{G,G} = (k-G)^2
            # So D^{-1} M E = E
            # Or: (D^{-1} M - I) E = 0
            
            # Compute D
            D = np.zeros((n_g, n_g))
            for ig, g in enumerate(g_vectors):
                kg = k - np.array(g) * 2 * math.pi
                D[ig, ig] = kg[0]**2 + kg[1]**2
            
            # Avoid division by zero
            D_inv = np.zeros_like(D)
            for ig in range(n_g):
                if abs(D[ig, ig]) > 1e-10:
                    D_inv[ig, ig] = 1.0 / D[ig, ig]
            
            # A = D^{-1} M
            A = D_inv @ M
            
            # Find eigenvalues of A
            eigs = np.linalg.eigvals(A)
            # Take real parts
            eigs_real = np.real(eigs)
            eigs_real = eigs_real[eigs_real > 0]
            eigs_real = np.sort(eigs_real)
            
            if len(eigs_real) >= 5:
                omega4 = math.sqrt(eigs_real[3])  # 4th band (0-indexed: 3)
                omega5 = math.sqrt(eigs_real[4])  # 5th band (0-indexed: 4)
                all_eigs.append((omega4, omega5))
        
        if len(all_eigs) < 3:
            return -1.0
        
        # Find max omega_4 and min omega_5
        max_omega4 = max(o4 for o4, o5 in all_eigs)
        min_omega5 = min(o5 for o4, o5 in all_eigs)
        
        if max_omega4 == 0:
            return 0.0
        
        ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return ratio
    
    # Initial best
    best_score = -1.0
    best_params = (8.5, 8.5, 4.5)
    
    # Coarse grid search
    # cx, cy in [2, 14] (within first quadrant), radius in [2, 8]
    cx_vals = np.arange(2.0, 14.0, 2.0)
    cy_vals = np.arange(2.0, 14.0, 2.0)
    rad_vals = np.arange(2.0, 8.0, 1.0)
    
    for cx in cx_vals:
        for cy in cy_vals:
            for rad in rad_vals:
                if time.time() > deadline:
                    break
                cell = make_cell(cx, cy, rad)
                score = plane_wave_gap(cell, n_pw=13, n_kpts=6)
                if score > best_score:
                    best_score = score
                    best_params = (cx, cy, rad)
        
        if time.time() > deadline:
            break
    
    # Fine search around best
    cx0, cy0, r0 = best_params
    cx_fine = np.arange(max(1.0, cx0-2), min(15.0, cx0+2), 0.5)
    cy_fine = np.arange(max(1.0, cy0-2), min(15.0, cy0+2), 0.5)
    rad_fine = np.arange(max(1.5, r0-1.5), min(9.0, r0+1.5), 0.25)
    
    for cx in cx_fine:
        for cy in cy_fine:
            for rad in rad_fine:
                if time.time() > deadline:
                    break
                cell = make_cell(cx, cy, rad)
                score = plane_wave_gap(cell, n_pw=15, n_kpts=7)
                if score > best_score:
                    best_score = score
                    best_params = (cx, cy, rad)
        
        if time.time() > deadline:
            break
    
    # Very fine search
    cx0, cy0, r0 = best_params
    cx_very_fine = np.arange(max(1.0, cx0-0.5), min(15.0, cx0+0.5), 0.1)
    cy_very_fine = np.arange(max(1.0, cy0-0.5), min(15.0, cy0+0.5), 0.1)
    rad_very_fine = np.arange(max(1.5, r0-0.3), min(9.0, r0+0.3), 0.05)
    
    for cx in cx_very_fine:
        for cy in cy_very_fine:
            for rad in rad_very_fine:
                if time.time() > deadline:
                    break
                cell = make_cell(cx, cy, rad)
                score = plane_wave_gap(cell, n_pw=17, n_kpts=8)
                if score > best_score:
                    best_score = score
                    best_params = (cx, cy, rad)
        
        if time.time() > deadline:
            break
    
    # Generate final cell
    cx_final, cy_final, rad_final = best_params
    cell_np = make_cell(cx_final, cy_final, rad_final)
    
    # Convert to nested list
    cell = [[int(cell_np[i][j]) for j in range(size)] for i in range(size)]
    
    return cell
# EVOLVE-BLOCK-END
