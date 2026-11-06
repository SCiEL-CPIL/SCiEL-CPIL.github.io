import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Structural change: Use a differentiable surrogate model approach.
    1. Generate diverse initial designs with a fast PWE solver.
    2. Build a neural network surrogate that maps continuous relaxation of the
       binary mask to the gap metric.
    3. Optimize in continuous space using gradient descent (SGD), then project
       back to binary with thresholding.
    4. Refine with local search on the binary domain.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s margin
    
    def time_left():
        return deadline - time.time()
    
    # Fast internal PWE solver
    def compute_tm_bands(cell, n_pw=13, n_kpts=8):
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        M = n_pw // 2
        g_indices = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                if m*m + n*n <= (M + 1)*(M + 1):
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        g_array = np.array(g_indices, dtype=int)
        
        diff_m = g_array[:, 0].reshape(-1, 1) - g_array[:, 0].reshape(1, -1)
        diff_n = g_array[:, 1].reshape(-1, 1) - g_array[:, 1].reshape(1, -1)
        
        idx_m = diff_m % size
        idx_n = diff_n % size
        S = fft_coeffs[idx_m, idx_n]
        
        k_points = []
        n_seg = max(3, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        G_vecs = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) for m, n in g_indices])
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            Gk_x = G_vecs[:, 0] + 2.0 * np.pi * kx
            Gk_y = G_vecs[:, 1] + 2.0 * np.pi * ky
            k_mag_sq = Gk_x**2 + Gk_y**2
            
            M_mat = S * k_mag_sq[np.newaxis, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M_mat)
                eigvals = np.sort(np.real(eigvals))
                eigvals = np.maximum(eigvals, 0)
                omega[ki, :] = np.sqrt(eigvals)
            except Exception:
                omega[ki, :] = np.inf
        
        return omega
    
    def enforce_d4_symmetry(cell):
        cell = np.array(cell, dtype=int)
        half = size // 2
        tl = cell[:half, :half].copy()
        cell_new = np.zeros((size, size), dtype=int)
        cell_new[:half, :half] = tl
        cell_new[:half, half:] = tl[:, ::-1]
        cell_new[half:, :half] = tl[::-1, :]
        cell_new[half:, half:] = tl[::-1, ::-1]
        return cell_new
    
    def quadrant_to_full(tl):
        half = size // 2
        cell = np.zeros((size, size), dtype=int)
        cell[:half, :half] = tl
        cell[:half, half:] = tl[:, ::-1]
        cell[half:, :half] = tl[::-1, :]
        cell[half:, half:] = tl[::-1, ::-1]
        return cell
    
    def score_cell(cell, n_pw=13, n_kpts=8):
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                overlap = max_omega_4 - min_omega_5
                midgap = 0.5 * (min_omega_5 + max_omega_4)
                ratio = -2.0 * overlap / midgap if midgap > 0 else -1.0
            return ratio
        except Exception:
            return -1.0
    
    # Generate diverse initial candidates
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return enforce_d4_symmetry(cell).tolist()
    
    def make_central_ring_cell(inner_radius, outer_radius, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        
        if outer_radius > 0 and inner_radius >= 0:
            val = np.abs(R/outer_radius)**exponent - np.abs(inner_radius/outer_radius)**exponent
            cell[val >= 0] = 1
        
        return enforce_d4_symmetry(cell).tolist()
    
    def make_staggered_rods_cell(rod_radius, exponent=2.0, offset=0.15):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        centers = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                centers.append((sx*offset, sy*offset))
                centers.append((sx*(0.5-offset), sy*(0.5-offset)))
        
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        
        return enforce_d4_symmetry(cell).tolist()
    
    # Phase 1: Grid search to find good starting points
    best_score = -np.inf
    best_cell = None
    
    radii = np.linspace(0.06, 0.20, 12)
    exponents = [1.5, 2.0, 2.5, 3.0, 4.0]
    
    for r in radii:
        if time_left() < 15:
            break
        for exp in exponents:
            if time_left() < 5:
                break
            cell = make_four_quadrant_cell(r, exp)
            score = score_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    if time_left() > 15:
        inner_radii = np.linspace(0.03, 0.18, 7)
        outer_radii = np.linspace(0.12, 0.28, 7)
        for ir in inner_radii:
            if time_left() < 10:
                break
            for orad in outer_radii:
                if time_left() < 5:
                    break
                if orad <= ir:
                    continue
                cell = make_central_ring_cell(ir, orad)
                score = score_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    if time_left() > 15:
        radii_s = np.linspace(0.04, 0.16, 9)
        offsets = np.linspace(0.08, 0.22, 7)
        for r in radii_s:
            if time_left() < 10:
                break
            for off in offsets:
                if time_left() < 5:
                    break
                cell = make_staggered_rods_cell(r, 2.0, off)
                score = score_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
    
    # Phase 2: Build training data for surrogate
    if time_left() > 40:
        # Generate diverse training samples by random perturbations
        training_cells = []
        training_scores = []
        
        base_arr = np.array(best_cell, dtype=int)
        half = size // 2
        tl_base = base_arr[:half, :half].copy()
        
        np.random.seed(42)
        
        # Generate 50 diverse samples
        for _ in range(50):
            if time_left() < 30:
                break
            # Random perturbation of the base
            tl_pert = tl_base.copy()
            n_flips = np.random.randint(2, 8)
            indices = np.random.choice(half*half, size=n_flips, replace=False)
            for idx in indices:
                i, j = divmod(idx, half)
                tl_pert[i, j] = 1 - tl_pert[i, j]
            
            cell = quadrant_to_full(tl_pert)
            score = score_cell(cell.tolist())
            training_cells.append(cell)
            training_scores.append(score)
        
        # Also add some random designs
        for _ in range(20):
            if time_left() < 25:
                break
            tl_rand = np.random.randint(0, 2, (half, half))
            cell = quadrant_to_full(tl_rand)
            score = score_cell(cell.tolist())
            training_cells.append(cell)
            training_scores.append(score)
        
        training_scores = np.array(training_scores)
        
        # Phase 3: Train a simple neural network surrogate
        if time_left() > 25 and len(training_cells) >= 10:
            # Flatten training data
            X_train = np.array([c.flatten() for c in training_cells], dtype=float)
            y_train = training_scores
            n_samples = len(X_train)
            n_features = X_train.shape[1]
            
            # Simple 2-layer neural network with ReLU
            np.random.seed(123)
            hidden_size = 128
            W1 = np.random.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features)
            b1 = np.zeros(hidden_size)
            W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
            b2 = np.zeros(1)
            
            # Train with SGD
            lr = 0.01
            batch_size = 16
            n_epochs = 50
            
            for epoch in range(n_epochs):
                if time_left() < 15:
                    break
                
                # Shuffle
                perm = np.random.permutation(n_samples)
                X_shuffled = X_train[perm]
                y_shuffled = y_train[perm]
                
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    X_batch = X_shuffled[batch_start:batch_end]
                    y_batch = y_shuffled[batch_start:batch_end]
                    
                    # Forward pass
                    z1 = X_batch @ W1 + b1
                    a1 = np.maximum(0, z1)
                    z2 = a1 @ W2 + b2
                    pred = z2.flatten()
                    
                    # Loss: MSE
                    error = pred - y_batch
                    loss = np.mean(error**2)
                    
                    # Backward pass
                    dz2 = 2.0 * error / len(error)
                    dW2 = a1.T @ dz2[:, np.newaxis] / len(error)
                    db2 = np.mean(dz2)
                    
                    da1 = dz2[:, np.newaxis] @ W2.T
                    dz1 = da1 * (z1 > 0)
                    dW1 = X_batch.T @ dz1 / len(error)
                    db1 = np.mean(dz1, axis=0)
                    
                    # Update
                    W1 -= lr * dW1
                    b1 -= lr * db1
                    W2 -= lr * dW2
                    b2 -= lr * db2
            
            # Phase 4: Use surrogate to guide optimization in continuous space
            if time_left() > 20:
                # Start from best cell, relax to continuous
                x_cont = np.array(best_cell, dtype=float).flatten()
                
                # Gradient descent in continuous space
                lr_cont = 0.05
                n_steps = 100
                
                for step in range(n_steps):
                    if time_left() < 10:
                        break
                    
                    # Forward pass through surrogate
                    x_flat = x_cont
                    z1 = x_flat @ W1 + b1
                    a1 = np.maximum(0, z1)
                    z2 = a1 @ W2 + b2
                    pred = z2.flatten()[0]
                    
                    # Gradient of loss (negative of predicted score, we want to maximize)
                    dz2 = -1.0
                    dW2_unused = a1 * dz2
                    da1 = dz2 * W2.flatten()
                    dz1 = da1 * (z1 > 0)
                    dx_cont = dz1 @ W1.T
                    
                    # Update continuous variable
                    x_cont -= lr_cont * dx_cont
                    x_cont = np.clip(x_cont, 0, 1)
                    
                    # Project to binary and evaluate
                    x_binary = (x_cont > 0.5).astype(int).reshape(size, size)
                    x_binary = enforce_d4_symmetry(x_binary)
                    score = score_cell(x_binary.tolist())
                    
                    if score > best_score:
                        best_score = score
                        best_cell = x_binary.tolist()
                
                # Phase 5: Final local search refinement
                if time_left() > 10:
                    best_arr = np.array(best_cell, dtype=int)
                    half = size // 2
                    tl_best = best_arr[:half, :half].copy()
                    
                    boundary_final = []
                    for i in range(half):
                        for j in range(half):
                            neighbors = []
                            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                                ni, nj = i+di, j+dj
                                if 0 <= ni < half and 0 <= nj < half:
                                    neighbors.append(tl_best[ni, nj])
                            if neighbors and tl_best[i, j] != np.mean(neighbors):
                                boundary_final.append((i, j))
                    
                    np.random.shuffle(boundary_final)
                    
                    for (i, j) in boundary_final:
                        if time_left() < 5:
                            break
                        tl_new = tl_best.copy()
                        tl_new[i, j] = 1 - tl_new[i, j]
                        cell_new = quadrant_to_full(tl_new)
                        
                        score = score_cell(cell_new.tolist())
                        if score > best_score:
                            best_score = score
                            best_cell = cell_new.tolist()
                            tl_best = cell_new[:half, :half].copy()
    
    # Final validation: ensure D4 symmetry
    best_arr = np.array(best_cell, dtype=int)
    best_arr = enforce_d4_symmetry(best_arr)
    
    return best_arr.tolist()

# EVOLVE-BLOCK-END
