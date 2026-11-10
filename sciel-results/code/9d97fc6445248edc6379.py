import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 230s wall-clock kill
    
    # Use 17x17 plane-wave basis to match evaluator (289 PWs)
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # 30 K points in irreducible BZ matching evaluator
    k_points = []
    for i in range(11):
        t = i / 10.0 * 0.5
        k_points.append([t, 0.0])
    for i in range(1, 11):
        t = i / 10.0 * 0.5
        k_points.append([0.5, t])
    for i in range(1, 11):
        t = i / 10.0 * 0.5
        k_points.append([t, t])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix: for each pair (i,j), the difference G_i - G_j
    # maps to a Fourier coefficient index. We store the flat index into the 2D FFT grid.
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    # The FFT grid has indices 0..size-1. A difference (dx, dy) corresponds to
    # index ((dx % size), (dy % size)) in the 2D array, but we need to map to
    # the flattened coefficient array. We'll store the 2D indices directly.
    diff_idx_2d = (dx % size, dy % size)  # tuple of 2D arrays
    diff_idx_flat = diff_idx_2d[0].ravel() * size + diff_idx_2d[1].ravel()
    # Also need to know which entries are valid (all should be valid since we use mod)
    valid_diffs = np.ones((N_pw, N_pw), dtype=bool)
    
    def get_fourier_eps_inv(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given cell as a flat array."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        return coeffs_2d.ravel()
    
    def evaluate(cell_arr):
        """Evaluate the gap-midgap ratio."""
        eps_inv_coeffs_flat = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k, dtype=float)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # H[i,j] = eps_inv_coeffs[diff(G_i - G_j)] * k_plus_G_sq[j]
            # Use the precomputed flat indices
            H = eps_inv_coeffs_flat[diff_idx_flat].reshape(N_pw, N_pw) * k_plus_G_sq[None, :]
            
            H = (H + H.T) / 2.0
            evals = np.linalg.eigvalsh(H)
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return 0.0
        return gap / midgap

    def make_symmetric(cell):
        """Enforce C4v symmetry about cell center by setting all orbit members to the same value."""
        N = size
        result = cell.copy()
        # Process each unique orbit once
        visited = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(N):
                if visited[i, j]:
                    continue
                # Get the full orbit of (i, j)
                orbit = set()
                for (a, b) in [(i, j), (j, N-1-i), (N-1-i, N-1-j), (N-1-j, i),
                               (i, N-1-j), (N-1-i, j), (j, i), (N-1-j, N-1-i)]:
                    orbit.add((a, b))
                # Set all to the value at the first position
                val = result[i, j]
                for (r, c) in orbit:
                    result[r, c] = val
                    visited[r, c] = True
        return result

    def generate_superellipse_cell(n_order, scale, hole_r=0.0):
        """Generate a superellipse-shaped dielectric with optional center hole.
        Superellipse: |x|^n + |y|^n <= scale^n
        This family includes circles (n=2), squares (n->inf), and intermediate shapes.
        """
        cell = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                val = abs(x)**n_order + abs(y)**n_order
                if val <= scale**n_order:
                    cell[i, j] = 1.0
                if hole_r > 0 and x*x + y*y < hole_r**2:
                    cell[i, j] = 0.0
        cell = cell.astype(int)
        return make_symmetric(cell)
    
    def generate_ring_cell(outer_r, inner_r):
        """Generate a ring-shaped dielectric."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x*x + y*y
                if inner_r**2 <= r2 < outer_r**2:
                    cell[i, j] = 1
        return make_symmetric(cell)
    
    def generate_cross_cell(arm_width, arm_length, hole_radius=0.0):
        """Generate a cross/plus-shaped dielectric with optional center hole."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < arm_width/2 and abs(y) < arm_length/2:
                    cell[i, j] = 1
                if abs(y) < arm_width/2 and abs(x) < arm_length/2:
                    cell[i, j] = 1
                if hole_radius > 0:
                    if x*x + y*y < hole_radius**2:
                        cell[i, j] = 0
        return make_symmetric(cell)
    
    # Parameterized search
    best_score = -float('inf')
    best_cell = None
    
    def try_cell(cell):
        nonlocal best_score, best_cell
        if time.time() - start_time > time_limit:
            return
        score = evaluate(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Phase 1: Superellipse designs (structural change per knowledge)
    n_orders = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
    scales = [0.20, 0.25, 0.30, 0.35, 0.40]
    hole_rs = [0.0, 0.03, 0.05, 0.08]
    
    for no in n_orders:
        for sc in scales:
            for hr in hole_rs:
                if time.time() - start_time > time_limit:
                    break
                cell = generate_superellipse_cell(no, sc, hr)
                try_cell(cell)
    
    # Phase 2: Ring designs
    for outer in [0.25, 0.30, 0.35, 0.40]:
        for inner in [0.0, 0.05, 0.10, 0.15]:
            if time.time() - start_time > time_limit:
                break
            if inner < outer:
                cell = generate_ring_cell(outer, inner)
                try_cell(cell)
    
    # Phase 3: Cross designs
    for aw in [0.1, 0.15, 0.2, 0.25]:
        for al in [0.3, 0.4, 0.5]:
            if time.time() - start_time > time_limit:
                break
            for hr in [0.0, 0.03, 0.05, 0.08]:
                cell = generate_cross_cell(aw, al, hr)
                try_cell(cell)
    
    # Phase 4: Local perturbation refinement on best cell
    if best_cell is not None and time.time() - start_time < time_limit - 20:
        np.random.seed(42)
        for _ in range(500):
            if time.time() - start_time > time_limit:
                break
            cell = best_cell.copy()
            # Flip 1-3 symmetric orbits
            n_flips = np.random.randint(1, 4)
            for _ in range(n_flips):
                i = np.random.randint(size)
                j = np.random.randint(size)
                N = size
                orbit = set()
                for (a, b) in [(i, j), (j, N-1-i), (N-1-i, N-1-j), (N-1-j, i),
                               (i, N-1-j), (N-1-i, j), (j, i), (N-1-j, N-1-i)]:
                    orbit.add((a, b))
                for (r, c) in orbit:
                    cell[r, c] = 1 - cell[r, c]
            try_cell(cell)
    
    # Fallback
    if best_cell is None:
        best_cell = generate_ring_cell(0.30, 0.10)
    
    return best_cell.tolist()
