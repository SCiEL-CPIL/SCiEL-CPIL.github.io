"""QAOA MaxCut angle design via global dense-grid exhaustive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_c_vals(n, edges):
    """Precompute cost eigenvalues c_j for all basis states j."""
    dim = 1 << n
    c_vals = np.zeros(dim, dtype=np.float64)
    # Build a lookup: for each state j, count how many edges have same/diff bits
    # Vectorize: for each edge (u,v), compute whether bits differ for all j
    for u, v in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        # For all j in 0..dim-1, check if bit u == bit v
        # j & mask_u != 0  XOR  j & mask_v != 0  means different
        js = np.arange(dim, dtype=np.int64)
        diff = ((js & mask_u) != (js & mask_v)).astype(np.float64)
        # Z_u Z_v = -1 if different, +1 if same
        c_vals += 1.0 - 2.0 * diff
    return c_vals

def _expected_cut_fast(n, edges, gammas, betas, c_vals, edge_masks):
    """
    Fast expected cut using precomputed data.
    edge_masks: list of (mask_u, mask_v) for each edge.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    for k in range(len(gammas)):
        # Apply cost: diagonal phase
        psi *= np.exp(-1j * gammas[k] * c_vals)

        # Apply mixer: exp(-i beta * X_total)
        # X_total |j> = sum_i |j ^ (1<<i)>
        # We can compute this via bit-flip permutations
        x_psi = np.zeros_like(psi)
        for i in range(n):
            mask = 1 << i
            # For each j, x_psi[j] += psi[j ^ mask]
            # Vectorized: x_psi += psi with bit i flipped
            # Create index permutation
            js = np.arange(dim, dtype=np.int64)
            flipped = js ^ mask
            x_psi += psi[flipped]

        psi = np.cos(betas[k]) * psi - 1j * np.sin(betas[k]) * x_psi
        # Normalize
        norm = np.sqrt(np.real(np.vdot(psi, psi)))
        if norm > 1e-15:
            psi /= norm

    # Compute expected cut
    probs = np.abs(psi) ** 2
    exp_cut = 0.0
    for mask_u, mask_v in edge_masks:
        js = np.arange(dim, dtype=np.int64)
        diff = ((js & mask_u) != (js & mask_v)).astype(np.float64)
        # E[Z_u Z_v] = sum_j probs[j] * (1 - 2*diff[j])
        exp_cut += np.sum(probs * (1.0 - 2.0 * diff))

    return exp_cut

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force for small n."""
    if n > 20:
        raise ValueError("Exact MaxCut only for small n")
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _grid_search(n, edges, p, c_vals, edge_masks, deadline):
    """
    Global dense-grid search over (gamma_0, gamma_1, beta_0, beta_1).
    QAOA angles have periodicity: gamma ~ [0, pi], beta ~ [0, 2*pi] (or [0, pi] with sign).
    Actually for MaxCut, the relevant ranges are gamma in [0, pi] and beta in [0, 2*pi].
    But due to symmetry, we can restrict to [0, pi] for both.
    
    We'll use a multi-resolution approach:
    - Coarse grid: 8 points per dimension -> 8^4 = 4096 evaluations
    - Fine grid around best: 12 points per dimension in a window -> 12^4 = 20736
    - Even finer: 15 points in smaller window -> 15^4 = 50625
    """
    start_time = time.time()
    
    best_obj = -np.inf
    best_params = None
    
    # Phase 1: Coarse grid
    # gamma in [0, pi], beta in [0, 2*pi]
    # Use 10 points per dim -> 10^4 = 10000
    n_coarse = 10
    gamma_vals = np.linspace(0, np.pi, n_coarse)
    beta_vals = np.linspace(0, 2 * np.pi, n_coarse)
    
    for g0 in gamma_vals:
        for g1 in gamma_vals:
            for b0 in beta_vals:
                if time.time() > deadline:
                    break
                for b1 in beta_vals:
                    if time.time() > deadline:
                        break
                    gammas = np.array([g0, g1])
                    betas = np.array([b0, b1])
                    obj = _expected_cut_fast(n, edges, gammas, betas, c_vals, edge_masks)
                    if obj > best_obj:
                        best_obj = obj
                        best_params = (g0, g1, b0, b1)
    
    if time.time() > deadline:
        if best_params is not None:
            return np.array([[best_params[0], best_params[2]], [best_params[1], best_params[3]]])
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Phase 2: Fine grid around best
    if best_params is not None:
        g0c, g1c, b0c, b1c = best_params
        # Window: +/- pi/5 around center
        half_w = np.pi / 5
        n_fine = 12
        gamma_fine = np.linspace(max(0, g0c - half_w), min(np.pi, g0c + half_w), n_fine)
        gamma_fine_2 = np.linspace(max(0, g1c - half_w), min(np.pi, g1c + half_w), n_fine)
        beta_fine = np.linspace(max(0, b0c - half_w), min(2*np.pi, b0c + half_w), n_fine)
        beta_fine_2 = np.linspace(max(0, b1c - half_w), min(2*np.pi, b1c + half_w), n_fine)
        
        for g0 in gamma_fine:
            for g1 in gamma_fine_2:
                for b0 in beta_fine:
                    if time.time() > deadline:
                        break
                    for b1 in beta_fine_2:
                        if time.time() > deadline:
                            break
                        gammas = np.array([g0, g1])
                        betas = np.array([b0, b1])
                        obj = _expected_cut_fast(n, edges, gammas, betas, c_vals, edge_masks)
                        if obj > best_obj:
                            best_obj = obj
                            best_params = (g0, g1, b0, b1)
        
        if time.time() > deadline:
            return np.array([[best_params[0], best_params[2]], [best_params[1], best_params[3]]])
        
        # Phase 3: Even finer
        if best_params is not None:
            g0c, g1c, b0c, b1c = best_params
            half_w2 = np.pi / 15
            n_fine2 = 15
            gamma_fine2 = np.linspace(max(0, g0c - half_w2), min(np.pi, g0c + half_w2), n_fine2)
            gamma_fine2_2 = np.linspace(max(0, g1c - half_w2), min(np.pi, g1c + half_w2), n_fine2)
            beta_fine2 = np.linspace(max(0, b0c - half_w2), min(2*np.pi, b0c + half_w2), n_fine2)
            beta_fine2_2 = np.linspace(max(0, b1c - half_w2), min(2*np.pi, b1c + half_w2), n_fine2)
            
            for g0 in gamma_fine2:
                for g1 in gamma_fine2_2:
                    for b0 in beta_fine2:
                        if time.time() > deadline:
                            break
                        for b1 in beta_fine2_2:
                            if time.time() > deadline:
                                break
                            gammas = np.array([g0, g1])
                            betas = np.array([b0, b1])
                            obj = _expected_cut_fast(n, edges, gammas, betas, c_vals, edge_masks)
                            if obj > best_obj:
                                best_obj = obj
                                best_params = (g0, g1, b0, b1)
    
    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    return np.array([[best_params[0], best_params[2]], [best_params[1], best_params[3]]])

def angles(n, edges, p):
    """
    Entry point: design QAOA angles via global grid search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    try:
        # Precompute
        c_vals = _precompute_c_vals(n, edges)
        edge_masks = [(1 << u, 1 << v) for u, v in edges]
        
        # Run grid search
        result = _grid_search(n, edges, p, c_vals, edge_masks, deadline)
        return result
    except Exception:
        # Fallback
        return np.tile([0.35, 0.25], (p, 1))

# EVOLVE-BLOCK-END
