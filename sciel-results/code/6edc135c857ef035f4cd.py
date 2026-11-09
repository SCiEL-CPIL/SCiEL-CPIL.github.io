import numpy as np
import time
import itertools

def design(problem):
    """
    Designs a C4-symmetric 32x32 unit cell to maximize the TM band gap
    between bands 4 and 5 in a square-lattice photonic crystal.
    Uses a direct search on the 8x8 symmetric quadrant with a fast
    plane-wave evaluator.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # The unit cell is 32x32 with C4v symmetry about the center.
    # We parameterize the top-left 8x8 quadrant and reflect it.
    # The center of the 32x32 cell is at (15.5, 15.5) in pixel coords.
    # The quadrant is pixels [0:8, 0:8].
    # Reflection: cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
    #             and cell[i][j] = cell[j][i]
    
    # Precompute the mapping from quadrant pixel (i, j) to all 8 positions in the cell
    # For a 32x32 cell, the top-left quadrant is [0:16, 0:16] but with C4v symmetry
    # about the center, the fundamental domain is 8x8 if we consider the half-rotation.
    # Actually, for C4v symmetry of a 32x32 grid:
    # The symmetry group has 8 elements. The fundamental domain has 32*32/8 = 128 pixels.
    # But we can be smarter: the top-left 8x8 quadrant (pixels 0-7 in both x and y)
    # determines the entire cell under the full C4v group.
    
    # Let's define the quadrant as 8x8 pixels (indices 0-7).
    quad_size = size // 4  # 8
    
    def expand_quadrant(q):
        """Expand an 8x8 quadrant to a full 32x32 C4v-symmetric cell."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(quad_size):
            for j in range(quad_size):
                val = q[i, j]
                # 8 symmetric positions
                positions = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, i),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                for pi, pj in positions:
                    cell[pi, pj] = val
        return cell
    
    def get_epsilon_cell(q):
        """Convert quadrant to epsilon grid."""
        cell = expand_quadrant(q)
        return np.where(cell > 0.5, eps_high, eps_low)
    
    # Precompute plane-wave setup
    N_pw = 17  # 17x17 = 289 plane waves
    G_vals = np.arange(-N_pw // 2, N_pw // 2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # k-points in irreducible BZ (Gamma-M, M-X, Gamma-X)
    # Using a coarse set for speed, but enough to capture the gap
    k_points = []
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G differences
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    
    def evaluate_quadrant_fast(q):
        """
        Fast evaluation of the band gap for a given quadrant.
        Uses the 289-PW Galerkin method on the k-points.
        """
        eps_grid = get_epsilon_cell(q)
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Build coefficient matrix
            coeff = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dx = int(G_diff[i, j, 0])
                    dy = int(G_diff[i, j, 1])
                    ix = dx % size
                    iy = dy % size
                    coeff[i, j] = fft_inv_eps[ix, iy]
            
            M = coeff * omega_sq[:, np.newaxis]
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
        return 2 * gap / midgap
    
    # Optimization: Start with a few good initial guesses
    # Initial guess 1: Dielectric rods (high dielectric in center)
    q1 = np.zeros((quad_size, quad_size), dtype=int)
    q1[3:6, 3:6] = 1  # Center block
    q1[2:7, 2:7] = 1  # Larger center block
    
    # Initial guess 2: Air holes (low dielectric in center)
    q2 = np.ones((quad_size, quad_size), dtype=int)
    q2[3:5, 3:5] = 0  # Center hole
    
    # Initial guess 3: Checkerboard-like
    q3 = np.zeros((quad_size, quad_size), dtype=int)
    for i in range(quad_size):
        for j in range(quad_size):
            if (i + j) % 2 == 0:
                q3[i, j] = 1
    
    # Initial guess 4: Ring-like
    q4 = np.zeros((quad_size, quad_size), dtype=int)
    q4[1:7, 1:7] = 1
    q4[3:5, 3:5] = 0
    
    best_q = None
    best_score = -np.inf
    
    for q_init in [q1, q2, q3, q4]:
        score = evaluate_quadrant_fast(q_init)
        if score > best_score:
            best_score = score
            best_q = q_init.copy()
    
    if best_q is None:
        # Fallback: return a simple symmetric pattern
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i % 4 < 2) == (j % 4 < 2):
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
    
    # Local search: Hill climbing on the 8x8 quadrant
    current_q = best_q.copy()
    current_score = best_score
    
    # Precompute neighbor list for each pixel in the quadrant
    neighbors = []
    for i in range(quad_size):
        for j in range(quad_size):
            nbrs = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < quad_size and 0 <= nj < quad_size:
                        nbrs.append((ni, nj))
            neighbors.append(nbrs)
    
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations and time.time() < deadline - 10:
        improved = False
        iteration += 1
        for idx in range(quad_size * quad_size):
            i, j = idx // quad_size, idx % quad_size
            # Try flipping this pixel
            old_val = current_q[i, j]
            current_q[i, j] = 1 - old_val
            new_score = evaluate_quadrant_fast(current_q)
            
            if new_score > current_score:
                current_score = new_score
                best_score = current_score
                best_q = current_q.copy()
                improved = True
            else:
                current_q[i, j] = old_val
    
    # Random restarts if time permits
    if time.time() < deadline - 20:
        num_restarts = 3
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            # Random quadrant
            q_random = np.random.randint(0, 2, (quad_size, quad_size))
            score = evaluate_quadrant_fast(q_random)
            if score > best_score:
                best_score = score
                best_q = q_random.copy()
            
            # Local refinement from random start
            q_local = q_random.copy()
            s_local = score
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                idx = np.random.randint(0, quad_size * quad_size)
                i, j = idx // quad_size, idx % quad_size
                old_val = q_local[i, j]
                q_local[i, j] = 1 - old_val
                new_s = evaluate_quadrant_fast(q_local)
                if new_s > s_local:
                    s_local = new_s
                    q_local = q_local.copy()
                    if s_local > best_score:
                        best_score = s_local
                        best_q = q_local.copy()
                else:
                    q_local[i, j] = old_val
    
    # Expand best quadrant to full cell
    final_cell = expand_quadrant(best_q)
    
    # Convert to list of lists of 0/1
    result = []
    for row in final_cell:
        result.append([int(x) for x in row])
    
    return result
