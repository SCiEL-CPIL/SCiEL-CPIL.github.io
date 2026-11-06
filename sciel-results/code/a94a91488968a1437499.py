import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s budget
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Symmetrize helper: Enforce C4v symmetry exactly
    # The problem requires: cell == cell[::-1] (rows), cell == cell[:, ::-1] (cols), cell == cell.T
    # This implies invariance under the square symmetry group.
    # We project any array onto this subspace by averaging over the 8 symmetries.
    def symmetrize(c):
        c1 = c
        c2 = c[:, ::-1]
        c3 = c[::-1, :]
        c4 = c[::-1, ::-1]
        c5 = c.T
        c6 = c.T[:, ::-1]
        c7 = c.T[::-1, :]
        c8 = c.T[::-1, ::-1]
        avg = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) / 8.0
        # Binarize: 0 or 1
        return (avg > 0.5).astype(np.float64)

    # 2. PWE Solver Setup
    # Use a sufficient number of plane waves for convergence.
    # 289 is the evaluator count (17x17). We use 15x15=225 for speed, which is usually sufficient for rough optimization.
    N_G = 15
    G_vecs = []
    for m in range(-N_G//2, N_G//2 + 1):
        for n in range(-N_G//2, N_G//2 + 1):
            G_vecs.append((2*math.pi*m, 2*math.pi*n))
    G_vecs = np.array(G_vecs)
    N_pws = len(G_vecs)
    G_dict = {tuple(g): i for i, g in enumerate(G_vecs)}
    
    # Precompute pixel positions
    r = (np.arange(size) + 0.5) / size - 0.5
    Rx, Ry = np.meshgrid(r, r, indexing='ij')

    def get_epsilon_fourier(cell_map):
        """Compute Fourier coefficients of epsilon(x) for the given G vectors."""
        eps_coeffs = np.zeros(N_pws, dtype=complex)
        eps_val = eps_low + (eps_high - eps_low) * cell_map
        
        for k_idx, G in enumerate(G_vecs):
            Gx, Gy = G
            phase = np.exp(-1j * (Gx * Rx + Gy * Ry))
            eps_coeffs[k_idx] = np.sum(eps_val * phase) / (size * size)
            
        return eps_coeffs

    def build_B_matrix(eps_coeffs):
        """Build the B matrix (epsilon coupling matrix)."""
        B = np.zeros((N_pws, N_pws), dtype=complex)
        for i, Gi in enumerate(G_vecs):
            for j, Gj in enumerate(G_vecs):
                diff = tuple(Gi - Gj)
                if diff in G_dict:
                    idx = G_dict[diff]
                    B[i, j] = eps_coeffs[idx]
        B = (B + B.conj().T) / 2.0
        return B

    def compute_bands(kx, ky, B_mat):
        """Solve the generalized eigenvalue problem for given k."""
        A_diag = np.zeros(N_pws)
        for i, G in enumerate(G_vecs):
            kx_g = kx + G[0]
            ky_g = ky + G[1]
            A_diag[i] = kx_g**2 + ky_g**2
            
        try:
            # Solve A v = w^2 B v  =>  B^{-1} A v = w^2 v? No, usually A v = w^2 B v.
            # scipy.linalg.eigh(a, b) solves a v = w b v.
            # Here a = A (diagonal), b = B.
            w2, v = np.linalg.eigh(A_diag, B_mat)
            idx_sort = np.argsort(w2)
            w2 = w2[idx_sort]
            v = v[:, idx_sort]
            w = np.sqrt(np.maximum(w2, 0))
            return w, v
        except:
            return None, None

    def evaluate_gap(cell_map):
        """Evaluate the gap-midgap ratio for a given cell."""
        eps_c = get_epsilon_fourier(cell_map)
        B_mat = build_B_matrix(eps_c)
        
        # Key high-symmetry points in the irreducible BZ
        # Gamma (0,0), X (pi, 0), M (pi, pi)
        # The gap between band 4 and 5 usually opens at M or X for square lattices.
        # We check all three to be safe.
        k_points = [
            (0.0, 0.0),      # Gamma
            (math.pi, 0.0),  # X
            (math.pi, math.pi) # M
        ]
        
        max_w4 = -1
        min_w5 = 1e9
        best_k = None
        best_v4 = None
        best_v5 = None
        
        # First pass: find global max w4 and min w5
        for kx, ky in k_points:
            w, v = compute_bands(kx, ky, B_mat)
            if w is None:
                continue
            if len(w) >= 5:
                w4 = w[3]
                w5 = w[4]
                if w4 > max_w4:
                    max_w4 = w4
                    # Store v4 for this k (might need to re-fetch if max changes later, but for now store)
                    # Actually, we need v4 and v5 at the SAME k for the mask construction?
                    # No, the mask is constructed from the difference.
                    # Ideally, we want the k where the gap is largest.
                    # Let's find the k that maximizes the gap directly.
        
        # Second pass: find k that maximizes gap = w5 - w4
        max_gap = -1
        best_gap_k = None
        best_v4 = None
        best_v5 = None
        best_w4 = 0
        best_w5 = 0
        
        for kx, ky in k_points:
            w, v = compute_bands(kx, ky, B_mat)
            if w is None:
                continue
            if len(w) >= 5:
                w4 = w[3]
                w5 = w[4]
                gap = w5 - w4
                if gap > max_gap:
                    max_gap = gap
                    best_gap_k = (kx, ky)
                    best_v4 = v[:, 3]
                    best_v5 = v[:, 4]
                    best_w4 = w4
                    best_w5 = w5
        
        if max_gap <= 0:
            return 0.0, None, None, None
            
        ratio = 2 * max_gap / (best_w5 + best_w4)
        return ratio, best_gap_k, best_v4, best_v5

    def get_field_intensities(v_coeff, kx, ky):
        """Compute |E(r)|^2 for a given mode vector."""
        field = np.zeros((size, size), dtype=complex)
        for i, G in enumerate(G_vecs):
            kx_g = kx + G[0]
            ky_g = ky + G[1]
            field += v_coeff[i] * np.exp(1j * (kx_g * Rx + ky_g * Ry))
        return np.abs(field)**2

    # 3. Optimization Loop using Band-Edge Mode Subtraction
    
    # Initial guess: A simple symmetric pattern. 
    # A checkerboard or a ring often works better than a single rod for higher band gaps.
    # Let's start with a "plus" shape or a ring to break degeneracies.
    # Actually, a random symmetric start might be better to avoid local minima of specific geometric priors.
    # But deterministic is safer. Let's use a ring.
    cell = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            dist = math.sqrt(x*x + y*y)
            # Ring between r=0.1 and r=0.3
            if 0.15 < dist < 0.35:
                cell[i, j] = 1.0
    cell = symmetrize(cell)
    
    best_score = 0.0
    best_cell = cell.copy()
    
    # Iterative Mode Subtraction
    # We will iterate:
    # 1. Evaluate bands.
    # 2. Extract |E4|^2 and |E5|^2 at the k-point of max gap.
    # 3. Construct target mask: M_target = |E5|^2 - |E4|^2.
    #    Where M_target > 0, we want dielectric (to push band 5 down).
    #    Where M_target < 0, we want air (to push band 4 up).
    # 4. Update cell by blending current cell with thresholded M_target.
    #    This is a form of "Direct Inverse Design" or "Mode Subtraction".
    
    num_iterations = 50
    blend_alpha = 0.5 # How strongly to move towards the target mask
    
    for it in range(num_iterations):
        if time.time() > deadline:
            break
            
        score, k_opt, v4, v5 = evaluate_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
        if k_opt is None or v4 is None or v5 is None:
            break
            
        kx, ky = k_opt
        
        # Get intensities
        int4 = get_field_intensities(v4, kx, ky)
        int5 = get_field_intensities(v5, kx, ky)
        
        # Normalize intensities to [0, 1] for stable thresholding
        max_int = max(np.max(int4), np.max(int5))
        if max_int > 0:
            int4 = int4 / max_int
            int5 = int5 / max_int
            
        # Construct the difference mask
        # We want high epsilon where int5 is high and int4 is low.
        # diff = int5 - int4
        diff = int5 - int4
        
        # Threshold the diff to create a binary target
        # If diff > 0, target is 1 (dielectric). If diff < 0, target is 0 (air).
        # To avoid flipping everything every step, we blend.
        target_mask = (diff > 0).astype(np.float64)
        
        # Symmetrize the target mask
        target_mask = symmetrize(target_mask)
        
        # Update cell: Blend current cell with target mask
        # cell_new = (1 - alpha) * cell + alpha * target_mask
        # Then threshold
        cell_float = (1 - blend_alpha) * cell + blend_alpha * target_mask
        cell = (cell_float > 0.5).astype(np.float64)
        cell = symmetrize(cell)
        
        # Optional: If score is very high, break early
        if best_score > 5.0:
            break

    # Final check: ensure best_cell is returned
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
        
    return result
