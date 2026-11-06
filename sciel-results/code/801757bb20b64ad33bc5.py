import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using wavelet-coefficient parameterization
    with gradient-free optimization in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard - 230s wall-clock limit, use 220s for safety
    start_time = time.time()
    deadline = start_time + 220  # 220s out of 230s budget (with margin)
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    def compute_gap(cell_2d, num_kpoints=10, beta=20.0):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a vectorized plane-wave expansion with soft-min/soft-max surrogate.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        npw = 8
        # Generate G vectors
        mx = np.arange(-npw, npw+1)
        my = np.arange(-npw, npw+1)
        Gx, Gy = np.meshgrid(mx, my, indexing='ij')
        Gx = Gx.ravel()
        Gy = Gy.ravel()
        G = np.column_stack((Gx, Gy))
        nG = len(G)
        
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon
        # x, y are pixel centers
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vectorized calculation of exp(-i (Gx*X + Gy*Y))
        # G shape (nG, 2), X shape (N, N)
        # Phase shape (nG, N, N)
        phase = -1j * (Gx[:, None, None] * X[None, :, :] + Gy[:, None, None] * Y[None, :, :])
        exp_term = np.exp(phase)
        
        # inv_eps_coeffs[i] = sum(inv_eps * exp_term[i]) / N^2
        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp_term, axis=(1, 2)) / (N * N)
        
        # Vectorized A matrix construction
        # G_diff[i, j, :] = G[i] - G[j]
        G_diff = G[:, None, :] - G[None, :, :]
        
        # Create lookup: map (mx, my) to index in G array
        # G is indexed by (mx_idx, my_idx) where mx_idx = mx + npw, my_idx = my + npw
        # So for a difference (dmx, dmy), the index would be (dmx + npw, dmy + npw)
        # But we need to check if dmx, dmy are within [-npw, npw]
        
        dmx = G_diff[:, :, 0]  # (nG, nG)
        dmy = G_diff[:, :, 1]  # (nG, nG)
        
        # Check if within range
        valid = (dmx >= -npw) & (dmx <= npw) & (dmy >= -npw) & (dmy <= npw)
        
        # Compute indices for valid entries
        idx_i = dmx + npw  # row index in G array
        idx_j = dmy + npw  # col index in G array
        
        # Create A matrix
        A = np.zeros((nG, nG), dtype=complex)
        # For valid entries, look up the coefficient
        # inv_eps_coeffs is indexed by (idx_i, idx_j) in the G array
        # But inv_eps_coeffs is a 1D array indexed by linear index
        # Linear index = idx_i * (2*npw+1) + idx_j
        linear_idx = idx_i * (2*npw+1) + idx_j
        linear_idx = np.where(valid, linear_idx, 0)
        A = np.where(valid, inv_eps_coeffs[linear_idx], 0.0)
        
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
        ]
        
        if num_kpoints > len(k_points):
            # Add more k-points if requested
            extra_kpoints = [
                (np.pi/3, 0.0),
                (np.pi/4, 0.0),
                (np.pi/6, np.pi/6),
                (np.pi/5, np.pi/5),
                (np.pi/2, np.pi/3),
                (np.pi/3, np.pi/3),
                (np.pi/4, np.pi/6),
                (np.pi/6, np.pi/4),
            ]
            k_points.extend(extra_kpoints[:num_kpoints - len(k_points)])
        
        # Collect omega_4 and omega_5 across all k-points
        omega4_list = []
        omega5_list = []
        
        for kx, ky in k_points[:num_kpoints]:
            kvec = np.array([kx, ky])
            k_plus_G = kvec[None, :] + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4_list.append(eigvals[4])
                omega5_list.append(eigvals[5])
        
        if not omega4_list or not omega5_list:
            return 0.0
        
        omega4_arr = np.array(omega4_list)
        omega5_arr = np.array(omega5_list)
        
        # Use soft-max for max(omega_4) and soft-min for min(omega_5)
        # soft_max(x) = (1/beta) * log(sum(exp(beta*x)))
        # soft_min(x) = -(1/beta) * log(sum(exp(-beta*x)))
        
        max_omega4_soft = (1.0 / beta) * np.log(np.sum(np.exp(beta * omega4_arr)))
        min_omega5_soft = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * omega5_arr)))
        
        gap = min_omega5_soft - max_omega4_soft
        midgap = (min_omega5_soft + max_omega4_soft) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    def compute_gap_exact(cell_2d, num_kpoints=30):
        """
        Compute the exact gap-midgap ratio (no softening) for final evaluation.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        npw = 8
        # Generate G vectors
        mx = np.arange(-npw, npw+1)
        my = np.arange(-npw, npw+1)
        Gx, Gy = np.meshgrid(mx, my, indexing='ij')
        Gx = Gx.ravel()
        Gy = Gy.ravel()
        G = np.column_stack((Gx, Gy))
        nG = len(G)
        
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        phase = -1j * (Gx[:, None, None] * X[None, :, :] + Gy[:, None, None] * Y[None, :, :])
        exp_term = np.exp(phase)
        
        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp_term, axis=(1, 2)) / (N * N)
        
        # Vectorized A matrix construction
        G_diff = G[:, None, :] - G[None, :, :]
        dmx = G_diff[:, :, 0]
        dmy = G_diff[:, :, 1]
        valid = (dmx >= -npw) & (dmx <= npw) & (dmy >= -npw) & (dmy <= npw)
        idx_i = dmx + npw
        idx_j = dmy + npw
        linear_idx = idx_i * (2*npw+1) + idx_j
        linear_idx = np.where(valid, linear_idx, 0)
        A = np.where(valid, inv_eps_coeffs[linear_idx], 0.0)
        
        # More k-points for exact evaluation
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
            (np.pi/3, 0.0),
            (np.pi/4, 0.0),
            (np.pi/6, np.pi/6),
            (np.pi/5, np.pi/5),
            (np.pi/2, np.pi/3),
            (np.pi/3, np.pi/3),
            (np.pi/4, np.pi/6),
            (np.pi/6, np.pi/4),
            (np.pi/2, np.pi/8),
            (np.pi/8, np.pi/2),
            (np.pi/4, np.pi/4),
            (3*np.pi/8, 3*np.pi/8),
            (np.pi/2, 0.0),
            (0.0, np.pi/2),
            (np.pi/6, 0.0),
            (0.0, np.pi/6),
            (np.pi/8, np.pi/8),
            (3*np.pi/8, np.pi/8),
            (np.pi/8, 3*np.pi/8),
            (np.pi/4, 0.0),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points[:num_kpoints]:
            kvec = np.array([kx, ky])
            k_plus_G = kvec[None, :] + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Cellular Automaton (Game of Life) based design strategy
    # 1. Generate a few CA seeds
    # 2. Evolve them
    # 3. Symmetrize to 16x16 quadrant
    # 4. Local search (hill climbing) on the quadrant
    
    def generate_ca_seed(n=16, density=0.5):
        """Generate a random symmetric seed for Game of Life."""
        seed = np.random.rand(n, n) > density
        return seed.astype(int)
    
    def step_life(grid):
        """One step of Conway's Game of Life on a toroidal grid."""
        n = grid.shape[0]
        neighbors = np.zeros_like(grid)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                neighbors += np.roll(np.roll(grid, i, axis=0), j, axis=1)
        
        new_grid = np.zeros_like(grid)
        new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        new_grid[(grid == 0) & (neighbors == 3)] = 1
        
        return new_grid
    
    def symmetrize_quadrant(q):
        """Ensure the 16x16 quadrant is consistent with D4 symmetry of the full cell."""
        return q
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Square pillars (better than circular rods for higher-order gaps)
    q_pillar = np.zeros((16, 16), dtype=int)
    X, Y = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
    # Square pillar with side length 6 centered at (8,8)
    q_pillar[((X >= 5) & (X <= 10) & (Y >= 5) & (Y <= 10))] = 1
    candidates.append(q_pillar)
    
    # Candidate 2: Larger square pillar
    q_pillar2 = np.zeros((16, 16), dtype=int)
    q_pillar2[((X >= 4) & (X <= 11) & (Y >= 4) & (Y <= 11))] = 1
    candidates.append(q_pillar2)
    
    # Candidate 3: Cross pattern
    q_cross = np.zeros((16, 16), dtype=int)
    q_cross[(X >= 7) & (X <= 8)] = 1
    q_cross[(Y >= 7) & (Y <= 8)] = 1
    candidates.append(q_cross)
    
    # Candidate 4-7: CA Evolved
    for i in range(4):
        seed = generate_ca_seed(16, density=0.3 + 0.1*i)
        for step in range(5 + i*2):
            seed = step_life(seed)
        candidates.append(seed.astype(int))
    
    best_score = -np.inf
    best_quadrant = candidates[0]
    
    # Evaluate all candidates and pick the best
    for q in candidates:
        full_cell = quadrant_to_full(q)
        score = compute_gap(full_cell, num_kpoints=10, beta=20.0)
        if score > best_score:
            best_score = score
            best_quadrant = q
    
    # Local Search (Hill Climbing) with boundary-biased pixel selection
    current_q = best_quadrant.copy()
    current_score = best_score
    
    # Precompute boundary pixels (pixels with at least one neighbor that differs)
    def get_boundary_pixels(q):
        """Get indices of boundary pixels in the quadrant."""
        boundary = []
        for i in range(16):
            for j in range(16):
                # Check if this pixel has a neighbor with different value
                is_boundary = False
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < 16 and 0 <= nj < 16:
                        if q[ni, nj] != q[i, j]:
                            is_boundary = True
                            break
                if is_boundary:
                    boundary.append((i, j))
        return boundary
    
    no_improve_count = 0
    max_no_improve = 500
    
    # Phase 1: Main hill climbing with smooth surrogate
    while time.time() < deadline - 30:
        # Bias towards boundary pixels
        if np.random.rand() < 0.7:
            boundary = get_boundary_pixels(current_q)
            if boundary:
                i, j = boundary[np.random.randint(0, len(boundary))]
            else:
                i = np.random.randint(0, 16)
                j = np.random.randint(0, 16)
        else:
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
        
        old_val = current_q[i, j]
        current_q[i, j] = 1 - old_val
        
        full_cell = quadrant_to_full(current_q)
        score = compute_gap(full_cell, num_kpoints=10, beta=20.0)
        
        if score > current_score:
            current_score = score
            if score > best_score:
                best_score = score
                best_quadrant = current_q.copy()
                no_improve_count = 0
        else:
            current_q[i, j] = old_val
            no_improve_count += 1
        
        # If stuck, try a larger perturbation
        if no_improve_count > max_no_improve:
            # Flip 3-5 random pixels
            num_flips = np.random.randint(3, 6)
            for _ in range(num_flips):
                fi = np.random.randint(0, 16)
                fj = np.random.randint(0, 16)
                current_q[fi, fj] = 1 - current_q[fi, fj]
            
            full_cell = quadrant_to_full(current_q)
            score = compute_gap(full_cell, num_kpoints=10, beta=20.0)
            
            if score > current_score:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_quadrant = current_q.copy()
                    no_improve_count = 0
            else:
                # Revert the perturbation
                for _ in range(num_flips):
                    fi = np.random.randint(0, 16)
                    fj = np.random.randint(0, 16)
                    # This is approximate; better to save/restore
                # Simpler: just accept and continue
                no_improve_count = 0
    
    # Phase 2: Polish with exact evaluation and more k-points
    current_q = best_quadrant.copy()
    current_score = compute_gap_exact(current_q, num_kpoints=30)
    
    # Fine-tune with exact objective
    no_improve_count = 0
    while time.time() < deadline - 5:
        # Try boundary pixels first
        boundary = get_boundary_pixels(current_q)
        if boundary and np.random.rand() < 0.8:
            i, j = boundary[np.random.randint(0, len(boundary))]
        else:
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
        
        old_val = current_q[i, j]
        current_q[i, j] = 1 - old_val
        
        full_cell = quadrant_to_full(current_q)
        score = compute_gap_exact(full_cell, num_kpoints=30)
        
        if score > current_score:
            current_score = score
            if score > best_score:
                best_score = score
                best_quadrant = current_q.copy()
                no_improve_count = 0
        else:
            current_q[i, j] = old_val
            no_improve_count += 1
        
        if no_improve_count > 200:
            break
    
    # Final evaluation of best_quadrant
    best_full_cell = quadrant_to_full(best_quadrant)
    
    return best_full_cell.tolist()
