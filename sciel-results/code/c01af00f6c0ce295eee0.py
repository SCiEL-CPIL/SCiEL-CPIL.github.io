# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Minimizes union volume via separable 1D optimization with efficient evaluation.
    """
    deadline = time.time() + 580.0
    N = 8
    n = N
    
    def keich_positions(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich
    
    def union_length_1d_fast(c, positions):
        """Vectorized union length computation."""
        g_idx = np.arange(n)
        lefts = positions + g_idx * (1.0 + c) / N
        rights = positions + (g_idx + 1.0 + c * g_idx) / N
        order = np.argsort(lefts)
        lefts_s = lefts[order]
        rights_s = rights[order]
        # Vectorized union length
        total = 0.0
        current_end = lefts_s[0]
        total = rights_s[0] - lefts_s[0]
        for i in range(1, n):
            if lefts_s[i] > current_end:
                total += rights_s[i] - lefts_s[i]
                current_end = rights_s[i]
            else:
                if rights_s[i] > current_end:
                    total += rights_s[i] - current_end
                    current_end = rights_s[i]
        return total
    
    def compute_volume(x_pos, y_pos, n_c_samples=50):
        """Compute ∫ Ux(c)*Uy(c) dc using trapezoidal rule."""
        cs = np.linspace(0.0, 1.0, n_c_samples)
        ux = np.empty(n_c_samples)
        uy = np.empty(n_c_samples)
        for idx, c in enumerate(cs):
            ux[idx] = union_length_1d_fast(c, x_pos)
            uy[idx] = union_length_1d_fast(c, y_pos)
        return np.trapz(ux * uy, cs)
    
    # Initial positions from Keich construction
    x0 = keich_positions(N).copy()
    y0 = keich_positions(N).copy()
    
    # Use a coordinate descent / subgradient approach
    # The key observation: union_length_1d is piecewise linear in each position
    # with slope in {-1, 0, +1}
    
    def subgradient_1d(positions, c):
        """
        Compute subgradient of union_length w.r.t. positions at height c.
        For interval k, the contribution to union length depends on whether
        its left endpoint is the minimum of its connected component and
        its right endpoint is the maximum.
        """
        g_idx = np.arange(n)
        lefts = positions + g_idx * (1.0 + c) / N
        rights = positions + (g_idx + 1.0 + c * g_idx) / N
        
        order = np.argsort(lefts)
        lefts_s = lefts[order]
        rights_s = rights[order]
        
        grad = np.zeros(n)
        
        # Process intervals in order of left endpoint
        current_end = lefts_s[0]
        # The first interval's left endpoint is always a boundary
        orig_first = order[0]
        grad[orig_first] += 1.0
        
        # Track the rightmost interval in the current component
        current_max_right_idx = 0
        current_max_right = rights_s[0]
        
        for i in range(1, n):
            if lefts_s[i] > current_end:
                # New component starts
                # The previous component's rightmost interval loses its right boundary contribution
                if i > 1:
                    prev_max_orig = order[current_max_right_idx]
                    grad[prev_max_orig] -= 1.0
                # New component: this interval's left is a boundary
                orig_i = order[i]
                grad[orig_i] += 1.0
                current_end = rights_s[i]
                current_max_right_idx = i
                current_max_right = rights_s[i]
            else:
                # Same component
                if rights_s[i] > current_max_right:
                    # This interval becomes the new rightmost
                    # Previous rightmost loses its contribution
                    prev_max_orig = order[current_max_right_idx]
                    grad[prev_max_orig] -= 1.0
                    # New rightmost gains contribution
                    orig_i = order[i]
                    grad[orig_i] -= 1.0
                    current_max_right_idx = i
                    current_max_right = rights_s[i]
                # Note: if rights_s[i] <= current_max_right, no change
                # (interval is fully contained, contributes nothing to boundary)
        
        # Last interval in last component: its right endpoint is a boundary
        last_orig = order[current_max_right_idx]
        # Wait, I need to add the right boundary contribution for the last component
        # The rightmost interval in the last component has -1 contribution
        # But I already handled this in the loop above when it became the max
        # Actually, let me reconsider: the union length = sum of (right - left) for each component
        # d/dc of a component's length = d(right_max)/dc - d(left_min)/dc
        # So the gradient is: +1 for the interval with min left in each component
        #                      -1 for the interval with max right in each component
        
        # Let me redo this more carefully
        grad = np.zeros(n)
        current_end = lefts_s[0]
        comp_left_min_idx = 0  # index in sorted array
        comp_right_max_idx = 0
        comp_right_max_val = rights_s[0]
        
        for i in range(1, n):
            if lefts_s[i] > current_end:
                # Close current component
                grad[order[comp_left_min_idx]] += 1.0
                grad[order[comp_right_max_idx]] -= 1.0
                # Start new component
                comp_left_min_idx = i
                comp_right_max_idx = i
                comp_right_max_val = rights_s[i]
                current_end = rights_s[i]
            else:
                if rights_s[i] > comp_right_max_val:
                    comp_right_max_idx = i
                    comp_right_max_val = rights_s[i]
        
        # Close last component
        grad[order[comp_left_min_idx]] += 1.0
        grad[order[comp_right_max_idx]] -= 1.0
        
        return grad
    
    # Optimize using subgradient method
    x = x0.copy()
    y = y0.copy()
    best_x = x.copy()
    best_y = y.copy()
    best_vol = compute_volume(x, y)
    
    n_c_samples = 30
    step = 0.03
    patience = 0
    max_patience = 100
    rng = np.random.default_rng(999)
    
    while time.time() < deadline:
        # Compute subgradient by averaging over several c values
        c_samples = rng.uniform(0, 1, size=5)
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        for c in c_samples:
            grad_x += subgradient_1d(x, c)
            grad_y += subgradient_1d(y, c)
        grad_x /= len(c_samples)
        grad_y /= len(c_samples)
        
        # Normalize
        grad_norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
        if grad_norm < 1e-12:
            break
        
        # Take step
        new_x = x - step * grad_x
        new_y = y - step * grad_y
        
        new_vol = compute_volume(new_x, new_y, n_c_samples)
        
        if new_vol < best_vol:
            best_x = new_x.copy()
            best_y = new_y.copy()
            best_vol = new_vol
            patience = 0
            step = min(step * 1.05, 0.15)
        else:
            patience += 1
            step *= 0.9
            if step < 1e-6:
                break
        
        x = new_x
        y = new_y
        
        if patience > max_patience:
            # Restart from best with perturbation
            x = best_x + rng.normal(0, 0.02, N)
            y = best_y + rng.normal(0, 0.02, N)
            step = 0.03
            patience = 0
    
    # Final polish: try to improve with smaller steps
    x = best_x.copy()
    y = best_y.copy()
    step = 0.01
    for _ in range(50):
        if time.time() > deadline:
            break
        c_samples = rng.uniform(0, 1, size=10)
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        for c in c_samples:
            grad_x += subgradient_1d(x, c)
            grad_y += subgradient_1d(y, c)
        grad_x /= len(c_samples)
        grad_y /= len(c_samples)
        
        new_x = x - step * grad_x
        new_y = y - step * grad_y
        new_vol = compute_volume(new_x, new_y, n_c_samples)
        
        if new_vol < best_vol:
            best_x = new_x.copy()
            best_y = new_y.copy()
            best_vol = new_vol
            x = new_x
            y = new_y
        else:
            step *= 0.8
    
    # Build output
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        base_x = x0[i_flat]
        base_y = y0[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
