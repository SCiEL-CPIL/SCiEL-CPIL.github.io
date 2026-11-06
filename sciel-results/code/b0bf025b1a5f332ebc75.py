import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # Precompute pixel coordinates for disk generation
    # Pixel centers are at (i + 0.5)
    # We will generate the cell by checking distance from disk centers
    
    # Disk centers for the 4x4 superlattice
    # Block size is 8x8. Centers at 4, 12, 20, 28.
    # Cell size 32.
    # Block (0,0) center: (4, 4)
    # Block (0,1) center: (12, 4)
    # Block (0,2) center: (20, 4)
    # Block (0,3) center: (28, 4)
    # ...
    
    # Symmetry orbits:
    # Corners: (4,4), (28,4), (4,28), (28,28)
    # Edges: (12,4), (20,4), (4,12), (4,20), (12,28), (20,28), (28,12), (28,20)
    # Inner: (12,12), (20,12), (12,20), (20,20)
    
    corner_centers = np.array([[4, 4], [28, 4], [4, 28], [28, 28]])
    edge_centers = np.array([[12, 4], [20, 4], [4, 12], [4, 20], [12, 28], [20, 28], [28, 12], [28, 20]])
    inner_centers = np.array([[12, 12], [20, 12], [12, 20], [20, 20]])
    
    # Pixel grid coordinates
    x = np.arange(size) + 0.5
    y = np.arange(size) + 0.5
    XX, YY = np.meshgrid(x, y, indexing='ij')
    
    def make_cell(r_c, r_e, r_i):
        """
        Create a 32x32 cell from disk radii.
        r_c: radius for corner disks
        r_e: radius for edge disks
        r_i: radius for inner disks
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Corner disks
        if r_c > 0:
            for cx, cy in corner_centers:
                dist = np.sqrt((XX - cx)**2 + (YY - cy)**2)
                cell[dist <= r_c] = 1
        
        # Edge disks
        if r_e > 0:
            for cx, cy in edge_centers:
                dist = np.sqrt((XX - cx)**2 + (YY - cy)**2)
                cell[dist <= r_e] = 1
                
        # Inner disks
        if r_i > 0:
            for cx, cy in inner_centers:
                dist = np.sqrt((XX - cx)**2 + (YY - cy)**2)
                cell[dist <= r_i] = 1
                
        return cell

    def calculate_gap(cell, G_max=5, k_points=None):
        """
        Calculate the band 4-5 gap ratio using the plane-wave method.
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        
        # G-vectors
        G_list = []
        for gx in range(-G_max, G_max + 1):
            for gy in range(-G_max, G_max + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        
        # Precompute inv_eps Fourier coefficients
        inv_eps_fft = np.fft.fft2(inv_eps)
        inv_eps_F = {}
        for gx, gy in G_list:
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[(gx, gy)] = inv_eps_fft[fft_gx, fft_gy] / (N * N)
        
        if k_points is None:
            # Irreducible BZ: 0 <= ky <= kx <= 0.5
            # We sample a few points
            k_points = [
                (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), # Gamma to X
                (0.5, 0.25), (0.5, 0.5), # X to M
                (0.25, 0.25) # Interior
            ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build eigenvalue matrix
            # M_{G,G'} = (G-k)·(G'-k) * inv_eps_F[G-G']
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio

    best_cell = None
    best_score = -float('inf')
    
    # Phase 1: Coarse Grid Search
    # Radii range [0, 4]
    # Step 0.5
    r_vals = np.arange(0, 4.1, 0.5)
    
    for r_c in r_vals:
        if time_left() < 10: break
        for r_e in r_vals:
            if time_left() < 10: break
            for r_i in r_vals:
                cell = make_cell(r_c, r_e, r_i)
                score = calculate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
    
    # Phase 2: Local Refinement around best
    if best_cell is not None and time_left() > 20:
        # We need to recover the best parameters or just do a random local search
        # Since we don't store the best params, let's do a random local search in the 3D space
        # But we need a starting point. Let's assume the best was found in the grid.
        # We can re-evaluate the grid points near the best score? No, we didn't store them.
        # Let's just do a random search with a tighter range around the previous best?
        # Actually, we can store the best params in the loop.
        
        # Let's restart the search with a finer grid around the best found in Phase 1?
        # We didn't store the best params. Let's modify the loop to store them.
        pass
    
    # Let's redo the logic to store best params
    best_params = (0, 0, 0)
    
    # Phase 1: Coarse Grid Search
    for r_c in r_vals:
        if time_left() < 10: break
        for r_e in r_vals:
            if time_left() < 10: break
            for r_i in r_vals:
                cell = make_cell(r_c, r_e, r_i)
                score = calculate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    best_params = (r_c, r_e, r_i)
                    
    # Phase 2: Local Refinement
    if time_left() > 30:
        bc, be, bi = best_params
        # Fine grid around best
        # Step 0.1
        r_fine = np.arange(max(0, bc-1), min(4, bc+1)+0.1, 0.1)
        e_fine = np.arange(max(0, be-1), min(4, be+1)+0.1, 0.1)
        i_fine = np.arange(max(0, bi-1), min(4, bi+1)+0.1, 0.1)
        
        for r_c in r_fine:
            if time_left() < 10: break
            for r_e in e_fine:
                if time_left() < 10: break
                for r_i in i_fine:
                    cell = make_cell(r_c, r_e, r_i)
                    score = calculate_gap(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_params = (r_c, r_e, r_i)

    # Phase 3: Random search if time remains
    if time_left() > 20:
        np.random.seed(42)
        for _ in range(int(time_left() / 2)): # ~2s per eval estimate
            if time_left() < 10: break
            r_c = np.random.uniform(0, 4)
            r_e = np.random.uniform(0, 4)
            r_i = np.random.uniform(0, 4)
            cell = make_cell(r_c, r_e, r_i)
            score = calculate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_params = (r_c, r_e, r_i)
            
            # Local refine around random point
            if score > best_score - 0.1: # If promising
                for _ in range(5):
                    r_c2 = np.clip(r_c + np.random.normal(0, 0.2), 0, 4)
                    r_e2 = np.clip(r_e + np.random.normal(0, 0.2), 0, 4)
                    r_i2 = np.clip(r_i + np.random.normal(0, 0.2), 0, 4)
                    cell2 = make_cell(r_c2, r_e2, r_i2)
                    score2 = calculate_gap(cell2)
                    if score2 > best_score:
                        best_score = score2
                        best_cell = cell2.copy()
                        best_params = (r_c2, r_e2, r_i2)
                        r_c, r_e, r_i = r_c2, r_e2, r_i2

    if best_cell is None:
        # Fallback
        best_cell = make_cell(2.0, 2.0, 2.0)

    return best_cell.tolist()
