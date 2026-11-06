import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using a fast vectorized plane-wave solver and
    pixel-based local search on C4v-symmetric block templates.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 210  # 210s out of 230s budget (with margin)
    
    def rot90(mat):
        return np.rot90(mat)
    
    def rot180(mat):
        return np.rot90(mat, 2)
    
    def rot270(mat):
        return np.rot90(mat, 3)
        
    def build_cell_from_templates(T0, T1, T2, T3):
        """
        Construct a 32x32 C4v symmetric cell from 4 8x8 templates.
        T0: Orbit (0,0), (0,3), (3,3), (3,0)
        T1: Orbit (1,1), (1,2), (2,2), (2,1)
        T2: Orbit (0,1), (1,3), (3,2), (2,0)
        T3: Orbit (0,2), (2,3), (3,1), (1,0)
        """
        B = np.zeros((4, 4, 8, 8), dtype=int)
        
        # Orbit 0
        B[0,0] = T0
        B[0,3] = rot90(T0)
        B[3,3] = rot180(T0)
        B[3,0] = rot270(T0)
        
        # Orbit 1
        B[1,1] = T1
        B[1,2] = rot90(T1)
        B[2,2] = rot180(T1)
        B[2,1] = rot270(T1)
        
        # Orbit 2
        B[0,1] = T2
        B[1,3] = rot90(T2)
        B[3,2] = rot180(T2)
        B[2,0] = rot270(T2)
        
        # Orbit 3
        B[0,2] = T3
        B[2,3] = rot90(T3)
        B[3,1] = rot180(T3)
        B[1,0] = rot270(T3)
        
        # Assemble 32x32
        cell = np.zeros((32, 32), dtype=int)
        for i in range(4):
            for j in range(4):
                cell[i*8:(i+1)*8, j*8:(j+1)*8] = B[i, j]
        
        return cell

    # Precompute G vectors and related structures
    npw = 8
    G = []
    for mx in range(-npw, npw+1):
        for my in range(-npw, npw+1):
            G.append((2.0*np.pi*mx, 2.0*np.pi*my))
    G = np.array(G, dtype=np.float64)
    nG = len(G)
    
    # Build index map for G vectors
    G_index = {}
    for idx, (gx, gy) in enumerate(G):
        G_index[(gx, gy)] = idx
    
    # Precompute the G_diff index matrix
    diff_idx = np.full((nG, nG), -1, dtype=np.int32)
    for i in range(nG):
        for j in range(nG):
            key = (G[i,0] - G[j,0], G[i,1] - G[j,1])
            if key in G_index:
                diff_idx[i, j] = G_index[key]
    
    # K-points in the irreducible BZ
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
        (np.pi/4, 0.0),
        (np.pi/6, 0.0),
        (np.pi/2, np.pi/3),
        (np.pi/3, 0.0),
        (np.pi/4, np.pi/8),
        (3*np.pi/8, 0.0),
        (5*np.pi/12, np.pi/12),
        (np.pi/5, np.pi/10),
        (2*np.pi/7, np.pi/7),
        (np.pi/2, np.pi/5),
        (3*np.pi/10, np.pi/10),
        (np.pi/6, np.pi/12),
        (5*np.pi/16, np.pi/16),
        (7*np.pi/24, np.pi/24),
        (np.pi/7, np.pi/14),
        (3*np.pi/16, np.pi/16),
        (5*np.pi/24, np.pi/24),
        (np.pi/8, 0.0),
        (np.pi/2, np.pi/8),
        (3*np.pi/8, 3*np.pi/8),
    ]
    
    # Precompute pixel coordinates
    N = 32
    x_pix = (np.arange(N) + 0.5) / N
    y_pix = (np.arange(N) + 0.5) / N
    X_pix, Y_pix = np.meshgrid(x_pix, y_pix, indexing='ij')
    
    # Precompute exp(-i*G*pixel)
    exp_phase = np.zeros((nG, N, N), dtype=np.complex128)
    for idx in range(nG):
        exp_phase[idx] = np.exp(-1j * (G[idx,0] * X_pix + G[idx,1] * Y_pix))
    
    def compute_gap_fast(cell_2d):
        """
        Fast computation of the gap-midgap ratio using vectorized operations.
        """
        eps = np.where(cell_2d == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        inv_eps_coeffs = np.zeros(nG, dtype=np.complex128)
        for idx in range(nG):
            inv_eps_coeffs[idx] = np.sum(inv_eps * exp_phase[idx]) / (N * N)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            A = np.zeros((nG, nG), dtype=np.complex128)
            valid = diff_idx >= 0
            A[valid] = inv_eps_coeffs[diff_idx[valid]]
            
            H = A * kG_sq[np.newaxis, :]
            H = (H + H.conj().T) / 2.0
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                if omega4 > max_omega4:
                    max_omega4 = omega4
                if omega5 < min_omega5:
                    min_omega5 = omega5
        
        if min_omega5 == float('inf') or max_omega4 <= 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap

    # Initialize templates with a random pattern
    np.random.seed(42)
    T0 = np.random.randint(0, 2, (8, 8))
    T1 = np.random.randint(0, 2, (8, 8))
    T2 = np.random.randint(0, 2, (8, 8))
    T3 = np.random.randint(0, 2, (8, 8))
    
    current_templates = [T0, T1, T2, T3]
    current_cell = build_cell_from_templates(T0, T1, T2, T3)
    best_score = compute_gap_fast(current_cell)
    best_templates = [t.copy() for t in current_templates]
    
    # Local search
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        # Pick a random template and pixel
        t_idx = np.random.randint(0, 4)
        r = np.random.randint(0, 8)
        c = np.random.randint(0, 8)
        
        # Flip pixel
        current_templates[t_idx][r, c] = 1 - current_templates[t_idx][r, c]
        new_cell = build_cell_from_templates(*current_templates)
        new_score = compute_gap_fast(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_templates = [t.copy() for t in current_templates]
            no_improve = 0
        else:
            # Revert
            current_templates[t_idx][r, c] = 1 - current_templates[t_idx][r, c]
            no_improve += 1
            
            # If stuck, try a random restart or larger perturbation
            if no_improve > 50:
                # Perturb multiple pixels
                for _ in range(5):
                    t_idx_p = np.random.randint(0, 4)
                    r_p = np.random.randint(0, 8)
                    c_p = np.random.randint(0, 8)
                    current_templates[t_idx_p][r_p, c_p] = 1 - current_templates[t_idx_p][r_p, c_p]
                new_cell = build_cell_from_templates(*current_templates)
                new_score = compute_gap_fast(new_cell)
                if new_score > best_score:
                    best_score = new_score
                    best_templates = [t.copy() for t in current_templates]
                    no_improve = 0
                else:
                    # Revert all
                    for _ in range(5):
                        t_idx_p = np.random.randint(0, 4)
                        r_p = np.random.randint(0, 8)
                        c_p = np.random.randint(0, 8)
                        # Note: This revert logic is flawed if the random picks are different.
                        # Better to keep a copy before perturbation.
                        pass 
                    # Simple fix: just reset to best if perturbation failed
                    current_templates = [t.copy() for t in best_templates]
                    no_improve = 0

    final_cell = build_cell_from_templates(*best_templates)
    return final_cell.tolist()
