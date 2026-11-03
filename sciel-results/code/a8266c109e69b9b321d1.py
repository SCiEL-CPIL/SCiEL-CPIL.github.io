"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute maximum possible radii for each circle given positions.
    Uses iterative relaxation to ensure strict non-overlap and boundary containment.
    """
    # Initial radii: distance to boundary
    radii = np.minimum(
        np.minimum(pos[:, 0], pos[:, 1]),
        np.minimum(1.0 - pos[:, 0], 1.0 - pos[:, 1])
    )
    radii = np.maximum(radii, 0.0)
    
    # Iterative relaxation for non-overlap
    for _ in range(100):
        changed = False
        for i in range(n):
            # Check boundary
            r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            if radii[i] > r_bound:
                radii[i] = r_bound
                changed = True
            
            # Check neighbors
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(pos[i] - pos[j])
                # Max radius for i given j is (d - radii[j])
                # But we want to maximize sum, so we distribute the gap
                # If d < radii[i] + radii[j], we need to reduce
                if d < radii[i] + radii[j]:
                    # Reduce both to fit in gap d
                    # New radii should be proportional to current, scaled to fit d
                    # Or simply set radii[i] = (d - radii[j]) if positive
                    # Better: reduce both by half the excess
                    excess = radii[i] + radii[j] - d
                    if excess > 0:
                        # Reduce i by half excess, but not below 0
                        reduction = excess / 2.0
                        new_r_i = radii[i] - reduction
                        new_r_j = radii[j] - reduction
                        
                        # Ensure non-negative
                        if new_r_i < 0:
                            new_r_i = 0.0
                        if new_r_j < 0:
                            new_r_j = 0.0
                        
                        # If one goes to 0, the other can take the rest?
                        # No, we must maintain d >= r_i + r_j
                        # If new_r_i + new_r_j > d, we still have overlap
                        # Let's just set them to fit exactly if possible
                        # Simpler: set r_i = max(0, (d - r_j)/2) and r_j = max(0, (d - r_i)/2) iteratively?
                        # Let's use a simple projection:
                        # r_i_new = max(0, radii[i] - excess/2)
                        # r_j_new = max(0, radii[j] - excess/2)
                        # If still overlapping, reduce more.
                        
                        radii[i] = max(0.0, radii[i] - excess / 2.0)
                        radii[j] = max(0.0, radii[j] - excess / 2.0)
                        changed = True
        
        if not changed:
            break
    
    # Final strict validation and correction
    for _ in range(10):
        changed = False
        for i in range(n):
            r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            if radii[i] > r_bound:
                radii[i] = r_bound
                changed = True
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    # Reduce both
                    radii[i] = max(0.0, radii[i] - excess / 2.0)
                    radii[j] = max(0.0, radii[j] - excess / 2.0)
                    changed = True
        if not changed:
            break
    
    return radii

def _discrete_best_fit(pos, n, deadline):
    """
    Discrete best-fit local search.
    For each circle, find the position on a local grid that maximizes its radius
    while maintaining non-overlap with others (whose radii are fixed during the step).
    """
    pos = pos.copy()
    radii = _compute_radii(pos, n)
    best_score = np.sum(radii)
    
    grid_resolution = 100  # 100x100 grid for local search
    
    while time.time() < deadline:
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Current radius of i
            r_i = radii[i]
            
            # Try to improve position of i
            # Search in a local neighborhood
            x0, y0 = pos[i]
            
            # Define search range
            search_range = 0.1
            xs = np.linspace(max(0, x0 - search_range), min(1, x0 + search_range), grid_resolution)
            ys = np.linspace(max(0, y0 - search_range), min(1, y0 + search_range), grid_resolution)
            
            best_new_r = r_i
            best_new_pos = pos[i].copy()
            
            for x in xs:
                for y in ys:
                    # Check boundary
                    r_bound = min(x, y, 1.0 - x, 1.0 - y)
                    if r_bound <= 0:
                        continue
                    
                    # Check overlap with others
                    # Max radius is limited by distance to others
                    min_dist_to_others = np.inf
                    for j in range(n):
                        if i == j:
                            continue
                        d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                        # Available radius is d - radii[j]
                        # But we also need to consider that radii[j] might change?
                        # In this greedy step, we fix others' radii.
                        avail_r = d - radii[j]
                        if avail_r < min_dist_to_others:
                            min_dist_to_others = avail_r
                    
                    # Max possible radius at (x, y)
                    r_max = min(r_bound, min_dist_to_others)
                    
                    if r_max > best_new_r + 1e-6:
                        best_new_r = r_max
                        best_new_pos = np.array([x, y])
            
            if best_new_r > r_i + 1e-6:
                # Update position and radius
                pos[i] = best_new_pos
                radii[i] = best_new_r
                best_score += best_new_r - r_i
                improved = True
                
                # Recompute other radii? No, that's expensive.
                # But moving i might allow others to grow?
                # Or might force others to shrink?
                # Since we only move if i can grow, and we keep others fixed,
                # others might now overlap with i's new larger radius?
                # Wait, best_new_r is calculated as min(dist to others - r_others, boundary).
                # So by definition, i's new circle does NOT overlap with others' current circles.
                # So others' radii remain valid.
                
        if not improved:
            break
    
    return pos, radii, best_score

def _initial_hexagonal(n):
    """Create initial hexagonal packing configuration."""
    pos = np.zeros((n, 2))
    spacing = 0.19
    row_spacing = spacing * np.sqrt(3) / 2
    idx = 0
    for row in range(8):
        for col in range(8):
            if idx >= n:
                break
            x = (col + 0.5) * spacing
            y = (row + 0.5) * row_spacing
            if row % 2 == 1:
                x += spacing / 2
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            pos[idx] = [x, y]
            idx += 1
    while idx < n:
        pos[idx] = [0.1 + 0.8 * np.random.rand(), 0.1 + 0.8 * np.random.rand()]
        idx += 1
    return pos

def _initial_grid(n):
    """Create initial grid configuration."""
    pos = np.zeros((n, 2))
    rows = 5
    cols = 6
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            pos[idx] = [x, y]
            idx += 1
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    return pos

def _initial_random(n, seed=42):
    """Create random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    min_sep = 0.08
    for i in range(n):
        attempts = 0
        while attempts < 100:
            p = rng.uniform(0.05, 0.95, size=2)
            ok = True
            for j in range(i):
                if np.linalg.norm(p - pos[j]) < min_sep:
                    ok = False
                    break
            if ok:
                pos[i] = p
                break
            attempts += 1
        if attempts >= 100:
            pos[i] = rng.uniform(0.05, 0.95, size=2)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initial configurations
    configs = []
    
    # Hexagonal
    pos_hex = _initial_hexagonal(N)
    configs.append(pos_hex)
    
    # Grid
    pos_grid = _initial_grid(N)
    configs.append(pos_grid)
    
    # Random starts
    for seed in [0, 1, 2, 3, 4]:
        pos_rand = _initial_random(N, seed=seed)
        configs.append(pos_rand)
    
    # Process each config
    for idx, pos_init in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        try:
            # Run discrete best-fit
            pos_opt, radii_opt, score = _discrete_best_fit(pos_init, N, deadline - 5)
            
            # Final validation
            # Re-compute radii strictly to ensure no overlap
            radii_final = _compute_radii(pos_opt, N)
            score_final = np.sum(radii_final)
            
            if score_final > best_score:
                best_score = score_final
                result = np.zeros((N, 3))
                result[:, :2] = pos_opt
                result[:, 2] = radii_final
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
    
    return best_result

# EVOLVE-BLOCK-END
