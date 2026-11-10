import numpy as np
import time

def design(problem):
    """
    Direct pixel-level optimization for TM band gap between bands 4 and 5.
    Uses a fast 289-PW plane-wave evaluator and stochastic local search.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute G vectors for 289 plane waves (N=15, so G from -15 to 15)
    N_pw = 15
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # 30 k-points in irreducible BZ (as used by evaluator)
    # Gamma to M: (0,0) to (0.5, 0)
    # M to X: (0.5, 0) to (0.5, 0.5)  
    # Gamma to X: (0,0) to (0.5, 0.5)
    k_points = []
    # Gamma-M: 10 points
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0))
    # M-X: 10 points
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # Gamma-X: 10 points
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    
    # Precompute G_diff for coefficient matrix
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using plane-wave method.
        eps_grid: 32x32 array of epsilon values
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Build coefficient matrix
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio
    
    def enforce_symmetry(cell):
        """Enforce C4v symmetry about cell center."""
        sym_cell = np.zeros((size, size), dtype=int)
        counts = np.zeros((size, size), dtype=int)
        
        for i in range(size):
            for j in range(size):
                val = cell[i, j]
                # All 8 symmetric positions under C4v
                positions = [
                    (i, j),
                    (size-1-i, j),       # reflect x
                    (i, size-1-j),       # reflect y
                    (size-1-i, size-1-j),# 180 rot
                    (j, i),              # transpose
                    (size-1-j, i),       # reflect y then transpose
                    (j, size-1-i),       # reflect x then transpose
                    (size-1-j, size-1-i) # 180 rot then transpose
                ]
                for pi, pj in positions:
                    sym_cell[pi, pj] += val
                    counts[pi, pj] += 1
        
        final_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if counts[i, j] > 0:
                    final_cell[i, j] = 1 if sym_cell[i, j] > counts[i, j] / 2.0 else 0
        return final_cell
    
    # Initial design: superlattice-inspired pattern
    # Use a checkerboard-like pattern with dielectric squares
    cell = np.zeros((size, size), dtype=int)
    # Create 4x4 superlattice of 8x8 blocks
    block_size = size // 4
    for bi in range(4):
        for bj in range(4):
            # Dielectric in certain blocks
            if (bi + bj) % 2 == 0:
                # Fill block with dielectric
                for i in range(bi * block_size, (bi + 1) * block_size):
                    for j in range(bj * block_size, (bj + 1) * block_size):
                        cell[i, j] = 1
    
    cell = enforce_symmetry(cell)
    
    # Convert to epsilon grid for evaluation
    eps_grid = np.where(cell == 1, eps_high, eps_low)
    best_score = evaluate_cell(eps_grid)
    best_cell = cell.copy()
    
    # Local search: flip pixels and check improvement
    np.random.seed(42)
    num_iterations = 0
    no_improve_count = 0
    
    while time.time() < deadline - 10:
        num_iterations += 1
        
        # Pick a random pixel to flip
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Flip the pixel and all its symmetric counterparts
        new_cell = best_cell.copy()
        positions = [
            (i, j),
            (size-1-i, j),
            (i, size-1-j),
            (size-1-i, size-1-j),
            (j, i),
            (size-1-j, i),
            (j, size-1-i),
            (size-1-j, size-1-i)
        ]
        
        old_vals = []
        for pi, pj in positions:
            old_vals.append(new_cell[pi, pj])
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_eps_grid = np.where(new_cell == 1, eps_high, eps_low)
        new_score = evaluate_cell(new_eps_grid)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Occasional restart from different initial patterns
        if no_improve_count > 500 and num_iterations % 500 == 0:
            # Try a few different initial patterns
            for pattern_idx in range(3):
                test_cell = np.zeros((size, size), dtype=int)
                if pattern_idx == 0:
                    # Concentric squares
                    for r in range(size // 2, 0, -1):
                        if r % 4 < 2:
                            for i in range(r, size - r):
                                for j in range(r, size - j if False else size - r):
                                    if abs(i - size//2 + 0.5) <= r and abs(j - size//2 + 0.5) <= r:
                                        if abs(i - size//2 + 0.5) >= r-1 or abs(j - size//2 + 0.5) >= r-1:
                                            test_cell[i, j] = 1
                elif pattern_idx == 1:
                    # Diagonal stripes
                    for i in range(size):
                        for j in range(size):
                            if (i + j) % 8 < 4:
                                test_cell[i, j] = 1
                else:
                    # Cross pattern
                    for i in range(size):
                        for j in range(size):
                            if abs(i - size//2) < 4 or abs(j - size//2) < 4:
                                test_cell[i, j] = 1
                
                test_cell = enforce_symmetry(test_cell)
                test_eps_grid = np.where(test_cell == 1, eps_high, eps_low)
                test_score = evaluate_cell(test_eps_grid)
                
                if test_score > best_score:
                    best_score = test_score
                    best_cell = test_cell.copy()
                    no_improve_count = 0
                    break
    
    # Final symmetry enforcement
    best_cell = enforce_symmetry(best_cell)
    
    # Return as nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
