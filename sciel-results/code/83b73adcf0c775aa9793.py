"""Photonic crystal design for maximum TM band gap (bands 4-5).
Strategy: Parametric geometric family (rings, crosses, holes) + coordinate descent
          with fast approximate band solver, final verification with full solver.
"""
import numpy as np
import time

def design(problem):
    start_time = time.time()
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    lower_band = problem.get("lower_band", 4)  # 4
    polarization = problem.get("polarization", "TM")
    
    # Deadline: leave 30s safety margin before 230s kill
    deadline = start_time + 200.0

    # --- Fast solver parameters (approximate, for search) ---
    N_fast = 6  # 13x13 = 169 plane waves
    G_range_fast = np.arange(-N_fast, N_fast + 1)
    Gx_f, Gy_f = np.meshgrid(G_range_fast, G_range_fast, indexing='ij')
    G_vectors_fast = np.column_stack((Gx_f.ravel(), Gy_f.ravel()))
    num_pw_fast = len(G_vectors_fast)
    
    # Fast k-points: 15 points in irreducible BZ wedge (0 <= ky <= kx <= 0.5)
    k_points_fast = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.25, 0.0), (0.25, 0.25), (0.5, 0.25),
        (0.125, 0.0), (0.375, 0.0), (0.125, 0.125), (0.375, 0.375),
        (0.5, 0.125), (0.5, 0.375),
        (0.2, 0.1), (0.4, 0.2), (0.3, 0.15),
    ]
    
    # --- Full solver parameters (for final verification) ---
    N_full = 8  # 17x17 = 289 plane waves
    G_range_full = np.arange(-N_full, N_full + 1)
    Gx_l, Gy_l = np.meshgrid(G_range_full, G_range_full, indexing='ij')
    G_vectors_full = np.column_stack((Gx_l.ravel(), Gy_l.ravel()))
    num_pw_full = len(G_vectors_full)
    
    # Full k-points: 30 points covering boundary and interior of irreducible BZ
    k_points_full = []
    # High-symmetry points
    k_points_full.append((0.0, 0.0))    # Gamma
    k_points_full.append((0.5, 0.0))    # X
    k_points_full.append((0.5, 0.5))    # M
    # Boundary: kx = 0.5, ky in [0, 0.5]
    for i in range(1, 6):
        k_points_full.append((0.5, 0.1 * i))
    # Boundary: ky = 0, kx in [0, 0.5]
    for i in range(1, 6):
        k_points_full.append((0.1 * i, 0.0))
    # Boundary: ky = kx (diagonal)
    for i in range(1, 6):
        k_points_full.append((0.1 * i, 0.1 * i))
    # Interior points
    interior = [
        (0.1, 0.05), (0.2, 0.1), (0.3, 0.15), (0.4, 0.2),
        (0.15, 0.05), (0.25, 0.1), (0.35, 0.15), (0.45, 0.25),
        (0.2, 0.05), (0.3, 0.1), (0.4, 0.15),
    ]
    k_points_full.extend(interior)
    # Deduplicate and ensure within wedge
    k_points_full = list(set(k_points_full))
    k_points_full = [(kx, ky) for kx, ky in k_points_full if 0 <= ky <= kx <= 0.5 + 1e-9]
    # Take up to 30
    k_points_full = k_points_full[:30]
    
    # Precompute lookup for full G vectors
    G_lookup_full = {}
    for idx in range(num_pw_full):
        key = (int(G_vectors_full[idx, 0]), int(G_vectors_full[idx, 1]))
        G_lookup_full[key] = idx
    
    # Precompute lookup for fast G vectors
    G_lookup_fast = {}
    for idx in range(num_pw_fast):
        key = (int(G_vectors_fast[idx, 0]), int(G_vectors_fast[idx, 1]))
        G_lookup_fast[key] = idx

    def compute_fourier_eps_inv_fast(cell):
        """Compute Fourier coefficients of 1/eps for fast solver."""
        inv_eps = np.where(cell == 0, 1.0, 1.0 / eps_high)
        inv_eps_G = np.zeros(num_pw_fast, dtype=complex)
        for idx in range(num_pw_fast):
            gx = G_vectors_fast[idx, 0]
            gy = G_vectors_fast[idx, 1]
            phase_x = np.exp(-2j * np.pi * gx * np.arange(size) / size)
            phase_y = np.exp(-2j * np.pi * gy * np.arange(size) / size)
            temp = inv_eps * phase_x[:, np.newaxis]
            temp = temp * phase_y[np.newaxis, :]
            inv_eps_G[idx] = np.sum(temp) / (size * size)
        return inv_eps_G

    def compute_fourier_eps_inv_full(cell):
        """Compute Fourier coefficients of 1/eps for full solver."""
        inv_eps = np.where(cell == 0, 1.0, 1.0 / eps_high)
        inv_eps_G = np.zeros(num_pw_full, dtype=complex)
        for idx in range(num_pw_full):
            gx = G_vectors_full[idx, 0]
            gy = G_vectors_full[idx, 1]
            phase_x = np.exp(-2j * np.pi * gx * np.arange(size) / size)
            phase_y = np.exp(-2j * np.pi * gy * np.arange(size) / size)
            temp = inv_eps * phase_x[:, np.newaxis]
            temp = temp * phase_y[np.newaxis, :]
            inv_eps_G[idx] = np.sum(temp) / (size * size)
        return inv_eps_G

    def build_hamiltonian(inv_eps_G, G_vectors, G_lookup, kx, ky):
        """Build the TM Hamiltonian matrix."""
        k_plus_G = G_vectors + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)
        n = len(G_vectors)
        H = np.zeros((n, n), dtype=complex)
        for a in range(n):
            ga = G_vectors[a]
            for b in range(n):
                gb = G_vectors[b]
                dgx = int(ga[0] - gb[0])
                dgy = int(ga[1] - gb[1])
                key = (dgx, dgy)
                if key in G_lookup:
                    inv_eps_val = inv_eps_G[G_lookup[key]]
                else:
                    inv_eps_val = 0.0
                H[a, b] = inv_eps_val * kG2[b]
        return H

    def evaluate_cell_fast(cell):
        """Fast evaluation using approximate solver."""
        inv_eps_G = compute_fourier_eps_inv_fast(cell)
        min_w5_sq = float('inf')
        max_w4_sq = 0.0
        for kx, ky in k_points_fast:
            H = build_hamiltonian(inv_eps_G, G_vectors_fast, G_lookup_fast, kx, ky)
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.real(eigenvalues)
            eigenvalues.sort()
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        if max_w4_sq <= 0:
            return 0.0
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        gap = min_w5 - max_w4
        return 2.0 * gap / midgap

    def evaluate_cell_full(cell):
        """Full evaluation using exact solver."""
        inv_eps_G = compute_fourier_eps_inv_full(cell)
        min_w5_sq = float('inf')
        max_w4_sq = 0.0
        for kx, ky in k_points_full:
            H = build_hamiltonian(inv_eps_G, G_vectors_full, G_lookup_full, kx, ky)
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.real(eigenvalues)
            eigenvalues.sort()
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        if max_w4_sq <= 0:
            return 0.0
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        gap = min_w5 - max_w4
        return 2.0 * gap / midgap

    # --- Parametric geometric design family ---
    # Parameters: 
    #   r_inner, r_outer: radii of central dielectric ring
    #   r_hole: radius of central air hole (if any)
    #   cross_width: width of cross arms
    #   cross_length: half-length of cross arms
    #   corner_hole_r: radius of corner holes
    #   mid_edge_hole_r: radius of mid-edge holes
    #   use_cross: 0 or 1
    #   use_corner_holes: 0 or 1
    #   use_mid_edge_holes: 0 or 1
    
    def generate_cell(params):
        """Generate a 32x32 cell from parametric description."""
        cell = np.zeros((size, size), dtype=np.uint8)
        center = (size - 1) / 2.0  # 15.5
        
        r_inner = params[0]
        r_outer = params[1]
        r_hole = params[2]
        cross_width = params[3]
        cross_length = params[4]
        corner_hole_r = params[5]
        mid_edge_hole_r = params[6]
        use_cross = params[7]
        use_corner_holes = params[8]
        use_mid_edge_holes = params[9]
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) - center
                y = (j + 0.5) - center
                r = np.sqrt(x * x + y * y)
                
                val = 0  # air by default
                
                # Central dielectric region: annulus between r_hole and r_outer
                if r_hole < r < r_outer:
                    val = 1
                
                # Cross arms (if enabled)
                if use_cross > 0.5:
                    # Horizontal arm: |y| < cross_width/2, |x| < cross_length
                    if abs(y) < cross_width / 2.0 and abs(x) < cross_length:
                        val = 1
                    # Vertical arm: |x| < cross_width/2, |y| < cross_length
                    if abs(x) < cross_width / 2.0 and abs(y) < cross_length:
                        val = 1
                
                # Corner holes (if enabled)
                if use_corner_holes > 0.5:
                    # Four corners at (±corner_dist, ±corner_dist)
                    corner_dist = size * 0.35
                    for cx in [corner_dist, -corner_dist]:
                        for cy in [corner_dist, -corner_dist]:
                            dc = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            if dc < corner_hole_r:
                                val = 0  # hole overrides dielectric
                
                # Mid-edge holes (if enabled)
                if use_mid_edge_holes > 0.5:
                    mid_dist = size * 0.4
                    for cx, cy in [(mid_dist, 0), (-mid_dist, 0), (0, mid_dist), (0, -mid_dist)]:
                        dc = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dc < mid_edge_hole_r:
                            val = 0
                
                cell[i, j] = val
        
        # Enforce square symmetry explicitly
        cell = np.maximum(cell, cell[::-1, :])  # vertical flip
        cell = np.maximum(cell, cell[:, ::-1])  # horizontal flip
        cell = np.maximum(cell, cell.T)         # transpose
        cell = np.maximum(cell, cell[::-1, ::-1])  # 180 rot
        # Actually, we need all 8 symmetries to be equal. Let's do it properly.
        # The above doesn't guarantee all 8. Let's regenerate with symmetry enforced.
        
        # Better approach: generate in fundamental domain and reflect
        cell2 = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                # Map to fundamental domain: 0 <= j <= i < size/2 roughly
                # Use the 8 symmetry operations to find the representative
                # For simplicity, just use the pixel value from the first generated cell
                # and enforce symmetry by averaging... but it's binary.
                # Let's just take the max over all 8 symmetric positions.
                positions = [
                    (i, j),
                    (size - 1 - i, size - 1 - j),
                    (i, size - 1 - j),
                    (size - 1 - i, j),
                    (j, i),
                    (size - 1 - j, size - 1 - i),
                    (size - 1 - i, size - 1 - j),  # duplicate
                    (j, size - 1 - i),
                ]
                val = 0
                for pi, pj in positions:
                    val = max(val, cell[pi, pj])
                cell2[i, j] = val
        return cell2

    # --- Initial parameter sets ---
    # Each param set: [r_inner, r_outer, r_hole, cross_width, cross_length, 
    #                  corner_hole_r, mid_edge_hole_r, use_cross, use_corner_holes, use_mid_edge_holes]
    initial_params = [
        # Ring only
        [5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [4.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [6.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        # Ring with central hole
        [5.0, 10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [4.0, 9.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [6.0, 11.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        # Ring with cross
        [5.0, 10.0, 0.0, 2.0, 12.0, 0.0, 0.0, 1, 0, 0],
        [4.0, 9.0, 0.0, 1.5, 11.0, 0.0, 0.0, 1, 0, 0],
        [6.0, 11.0, 0.0, 2.5, 13.0, 0.0, 0.0, 1, 0, 0],
        # Ring with corner holes
        [5.0, 10.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0, 1, 0],
        [4.0, 9.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0, 1, 0],
        # Ring with mid-edge holes
        [5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0, 0, 1],
        [4.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0, 0, 1],
        # Combined: ring + cross + corner holes
        [5.0, 10.0, 0.0, 2.0, 12.0, 2.0, 0.0, 1, 1, 0],
        [4.0, 9.0, 1.0, 1.5, 11.0, 1.5, 1.0, 1, 1, 1],
        # Larger ring
        [7.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [6.0, 11.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    ]
    
    # Evaluate initial params with fast solver
    best_score = -float('inf')
    best_params = None
    candidates = []
    
    for params in initial_params:
        cell = generate_cell(params)
        score = evaluate_cell_fast(cell)
        candidates.append((score, params.copy()))
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Sort candidates by fast score
    candidates.sort(key=lambda x: -x[0])
    
    # --- Coordinate descent on continuous parameters ---
    param_names = ['r_inner', 'r_outer', 'r_hole', 'cross_width', 'cross_length',
                   'corner_hole_r', 'mid_edge_hole_r', 'use_cross', 'use_corner_holes', 'use_mid_edge_holes']
    
    # Bounds for continuous params
    bounds = {
        'r_inner': (2.0, 14.0),
        'r_outer': (4.0, 15.0),
        'r_hole': (0.0, 8.0),
        'cross_width': (0.0, 6.0),
        'cross_length': (0.0, 16.0),
        'corner_hole_r': (0.0, 4.0),
        'mid_edge_hole_r': (0.0, 4.0),
    }
    
    def optimize_params(params, max_time=60.0):
        """Coordinate descent on continuous parameters."""
        start_opt = time.time()
        params = params.copy()
        current_score = evaluate_cell_fast(generate_cell(params))
        best_score_local = current_score
        best_params_local = params.copy()
        
        step_sizes = {
            'r_inner': 0.5,
            'r_outer': 0.5,
            'r_hole': 0.3,
            'cross_width': 0.3,
            'cross_length': 0.5,
            'corner_hole_r': 0.2,
            'mid_edge_hole_r': 0.2,
        }
        
        idx_map = {
            'r_inner': 0,
            'r_outer': 1,
            'r_hole': 2,
            'cross_width': 3,
            'cross_length': 4,
            'corner_hole_r': 5,
            'mid_edge_hole_r': 6,
        }
        
        while time.time() - start_opt < max_time:
            improved = False
            for pname in ['r_inner', 'r_outer', 'r_hole', 'cross_width', 'cross_length', 'corner_hole_r', 'mid_edge_hole_r']:
                idx = idx_map[pname]
                lo, hi = bounds[pname]
                step = step_sizes[pname]
                
                # Try positive and negative steps
                for direction in [1, -1]:
                    new_val = params[idx] + direction * step
                    if new_val < lo or new_val > hi:
                        continue
                    new_params = params.copy()
                    new_params[idx] = new_val
                    new_score = evaluate_cell_fast(generate_cell(new_params))
                    
                    if new_score > current_score + 1e-6:
                        params = new_params
                        current_score = new_score
                        improved = True
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_params_local = params.copy()
            
            # Also try flipping binary params
            for bidx in [7, 8, 9]:
                new_params = params.copy()
                new_params[bidx] = 1 - new_params[bidx]
                new_score = evaluate_cell_fast(generate_cell(new_params))
                if new_score > current_score + 1e-6:
                    params = new_params
                    current_score = new_score
                    improved = True
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_params_local = params.copy()
            
            if not improved:
                # Reduce step sizes
                for pname in step_sizes:
                    step_sizes[pname] *= 0.7
                # If steps are too small, break
                if all(step_sizes[p] < 0.05 for p in step_sizes):
                    break
            
            if time.time() - start_opt >= max_time:
                break
        
        return best_params_local, best_score_local

    # Optimize top candidates
    top_k = min(5, len(candidates))
    for i in range(top_k):
        if time.time() > deadline - 60:  # Leave 60s for final verification
            break
        score, params = candidates[i]
        opt_params, opt_score = optimize_params(params, max_time=30.0)
        if opt_score > best_score:
            best_score = opt_score
            best_params = opt_params.copy()
        candidates.append((opt_score, opt_params))
    
    # Re-sort and get top candidates for full verification
    candidates.sort(key=lambda x: -x[0])
    
    # --- Final verification with full solver ---
    # Take top few candidates and evaluate with full solver
    verified = []
    for i in range(min(3, len(candidates))):
        if time.time() > deadline - 30:
            break
        fast_score, params = candidates[i]
        cell = generate_cell(params)
        full_score = evaluate_cell_full(cell)
        verified.append((full_score, cell, params))
        print(f"Verification: fast={fast_score:.4f}, full={full_score:.4f}", flush=True)
    
    if verified:
        verified.sort(key=lambda x: -x[0])
        best_full_score, best_cell, best_params_final = verified[0]
    else:
        # Fallback: use best fast-scored cell
        best_cell = generate_cell(best_params)
        best_full_score = evaluate_cell_full(best_cell)
    
    # Return as nested list
    return best_cell.tolist()
