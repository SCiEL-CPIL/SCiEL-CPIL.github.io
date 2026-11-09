# EVOLVE-BLOCK-START
import time
import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 625  # leave 5s margin before 630s kill

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def objective(c, s):
        return np.sum(s)

    # --- Guillotine cut tree approach ---
    # Each node represents a rectangular region.
    # Leaf: contains one square (its side = min(width, height) of the region,
    #        centered in the region).
    # Internal: split horizontally or vertically at some ratio.
    
    def build_guillotine_tree(num_leaves, x0, y0, w, h, depth, max_depth,
                               split_ratios, best_config, best_val,
                               leaf_counter, config):
        """
        Recursively build a guillotine cut tree.
        Returns (best_val, best_config) for this subtree.
        """
        if time.time() > deadline:
            return best_val, best_config
        
        if num_leaves == 1:
            # Leaf: place one square in this rectangle
            s = min(w, h)
            if s <= 0:
                return best_val, best_config
            cx = x0 + w / 2
            cy = y0 + h / 2
            idx = leaf_counter[0]
            if idx < n:
                config['centers'][idx] = [cx, cy]
                config['sides'][idx] = s
                leaf_counter[0] += 1
            return best_val, best_config
        
        if depth >= max_depth:
            # Can't split further but still need to place squares
            # Just place all remaining squares as small as possible
            # This shouldn't happen with proper depth limits
            return best_val, best_config
        
        # Try horizontal and vertical splits
        for split_dir in [0, 1]:  # 0=horizontal (split width), 1=vertical (split height)
            for ratio in split_ratios:
                if time.time() > deadline:
                    return best_val, best_config
                
                if split_dir == 0:
                    # Horizontal split: left and right
                    w1 = w * ratio
                    w2 = w - w1
                    if w1 < 1e-9 or w2 < 1e-9:
                        continue
                    # Try all ways to distribute leaves between left and right
                    for n1 in range(1, num_leaves):
                        n2 = num_leaves - n1
                        # Left child
                        val1, cfg1 = build_guillotine_tree(
                            n1, x0, y0, w1, h, depth + 1, max_depth,
                            split_ratios, best_val, best_config, leaf_counter, config)
                        # Right child
                        val2, cfg2 = build_guillotine_tree(
                            n2, x0 + w1, y0, w2, h, depth + 1, max_depth,
                            split_ratios, best_val, best_config, leaf_counter, config)
                        total_val = val1 + val2
                        if total_val > best_val:
                            best_val = total_val
                            best_config = {k: v.copy() for k, v in config.items()}
                else:
                    # Vertical split: bottom and top
                    h1 = h * ratio
                    h2 = h - h1
                    if h1 < 1e-9 or h2 < 1e-9:
                        continue
                    for n1 in range(1, num_leaves):
                        n2 = num_leaves - n1
                        # Bottom child
                        val1, cfg1 = build_guillotine_tree(
                            n1, x0, y0, w, h1, depth + 1, max_depth,
                            split_ratios, best_val, best_config, leaf_counter, config)
                        # Top child
                        val2, cfg2 = build_guillotine_tree(
                            n2, x0, y0 + h1, w, h2, depth + 1, max_depth,
                            split_ratios, best_val, best_config, leaf_counter, config)
                        total_val = val1 + val2
                        if total_val > best_val:
                            best_val = total_val
                            best_config = {k: v.copy() for k, v in config.items()}
        
        return best_val, best_config

    def guillotine_search():
        """Search over guillotine cut trees to find good configurations."""
        # Split ratios to try: golden ratio, 1/2, 1/3, 2/3, etc.
        split_ratios = [0.5, 0.382, 0.618, 0.333, 0.667, 0.25, 0.75, 0.2, 0.8]
        max_depth = 5  # Limit depth to avoid explosion
        
        best_val = -1
        best_config = None
        
        # Try different initial leaf assignments
        for trial in range(50):
            if time.time() > deadline:
                break
            
            config = {
                'centers': np.zeros((n, 2)),
                'sides': np.zeros(n)
            }
            leaf_counter = [0]
            
            # Use a simplified recursive approach that tracks which square goes where
            # This is a heuristic: we'll use a simpler strategy
            # For now, just try a few fixed tree structures
            
            # Structure 1: Two large squares at bottom, 8 small at top
            # This is the analytical solution
            if trial == 0:
                centers = np.array([
                    [0.25, 0.25], [0.75, 0.25],
                    [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
                    [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
                ], dtype=np.float64)
                sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
                val = objective(centers, sides)
                if val > best_val:
                    best_val = val
                    best_config = (centers.copy(), sides.copy())
            
            # Structure 2: Random guillotine trees
            else:
                # Generate a random guillotine tree and compute the configuration
                # We'll use a recursive function that builds the tree and assigns squares
                def build_random_guillotine(num_leaves, x0, y0, w, h, depth, max_depth,
                                             config, leaf_counter):
                    if num_leaves == 1:
                        s = min(w, h)
                        if s <= 1e-9:
                            return 0
                        cx = x0 + w / 2
                        cy = y0 + h / 2
                        idx = leaf_counter[0]
                        if idx < n:
                            config['centers'][idx] = [cx, cy]
                            config['sides'][idx] = s
                            leaf_counter[0] += 1
                        return s
                    if depth >= max_depth:
                        # Force split
                        split_dir = np.random.randint(2)
                        ratio = np.random.uniform(0.3, 0.7)
                    else:
                        split_dir = np.random.randint(2)
                        ratio = np.random.uniform(0.3, 0.7)
                    
                    total_val = 0
                    if split_dir == 0:
                        w1 = w * ratio
                        w2 = w - w1
                        if w1 < 1e-9 or w2 < 1e-9:
                            return 0
                        n1 = np.random.randint(1, num_leaves)
                        n2 = num_leaves - n1
                        total_val += build_random_guillotine(n1, x0, y0, w1, h, depth+1, max_depth, config, leaf_counter)
                        total_val += build_random_guillotine(n2, x0+w1, y0, w2, h, depth+1, max_depth, config, leaf_counter)
                    else:
                        h1 = h * ratio
                        h2 = h - h1
                        if h1 < 1e-9 or h2 < 1e-9:
                            return 0
                        n1 = np.random.randint(1, num_leaves)
                        n2 = num_leaves - n1
                        total_val += build_random_guillotine(n1, x0, y0, w, h1, depth+1, max_depth, config, leaf_counter)
                        total_val += build_random_guillotine(n2, x0, y0+h1, w, h2, depth+1, max_depth, config, leaf_counter)
                    return total_val
                
                config = {
                    'centers': np.zeros((n, 2)),
                    'sides': np.zeros(n)
                }
                leaf_counter = [0]
                val = build_random_guillotine(n, 0, 0, 1, 1, 0, max_depth, config, leaf_counter)
                
                if leaf_counter[0] == n and val > best_val:
                    best_val = val
                    best_config = (config['centers'].copy(), config['sides'].copy())
        
        return best_config

    def coordinate_ascent(c, s, max_rounds=500):
        """
        Coordinate ascent: for each square, try to increase its side length
        by repositioning within feasible space.
        """
        c = c.copy()
        s = s.copy()
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                lo_s = s[i]
                hi_s = 1.0 + tol

                max_s_boundary = min(2 * c[i][0], 2 * (1 - c[i][0]),
                                     2 * c[i][1], 2 * (1 - c[i][1])) + 2 * tol
                hi_s = min(hi_s, max_s_boundary)
                if hi_s <= lo_s + 1e-12:
                    continue

                best_s = lo_s
                best_c = c[i].copy()

                for _ in range(40):
                    mid_s = (lo_s + hi_s) / 2
                    if mid_s <= best_s + 1e-13:
                        break

                    found = False
                    new_cx, new_cy = c[i][0], c[i][1]

                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid_s + s[j]) / 2 - tol
                        if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                            ok = False
                            break
                    if ok and new_cx - mid_s/2 >= -tol and new_cx + mid_s/2 <= 1+tol \
                       and new_cy - mid_s/2 >= -tol and new_cy + mid_s/2 <= 1+tol:
                        found = True
                    else:
                        for _ in range(200):
                            new_cx = np.random.uniform(mid_s/2, 1 - mid_s/2)
                            new_cy = np.random.uniform(mid_s/2, 1 - mid_s/2)
                            ok = True
                            for j in range(n):
                                if j == i:
                                    continue
                                t = (mid_s + s[j]) / 2 - tol
                                if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                break
                        if not found:
                            for dx in [0, 1]:
                                for dy in [0, 1]:
                                    new_cx = mid_s/2 + dx * (1 - mid_s)
                                    new_cy = mid_s/2 + dy * (1 - mid_s)
                                    ok = True
                                    for j in range(n):
                                        if j == i:
                                            continue
                                        t = (mid_s + s[j]) / 2 - tol
                                        if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                            ok = False
                                            break
                                    if ok:
                                        found = True
                                        break
                                if found:
                                    break

                    if found:
                        best_s = mid_s
                        best_c = np.array([new_cx, new_cy])
                        lo_s = mid_s
                    else:
                        hi_s = mid_s

                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    c[i] = best_c
                    improved = True

            if not improved:
                break

        return c, s

    def local_search(c, s, max_iters=2000):
        """More aggressive local search."""
        c = c.copy()
        s = s.copy()
        best_val = objective(c, s)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            i = np.random.randint(n)
            old_c = c[i].copy()
            old_s = s[i]

            ds = np.random.uniform(-0.05, 0.1)
            new_s = max(0.01, min(1.0, old_s + ds))
            dcx = np.random.uniform(-0.1, 0.1)
            dcy = np.random.uniform(-0.1, 0.1)
            new_cx = np.clip(old_c[0] + dcx, new_s/2, 1 - new_s/2)
            new_cy = np.clip(old_c[1] + dcy, new_s/2, 1 - new_s/2)

            c[i] = [new_cx, new_cy]
            s[i] = new_s

            if is_feasible(c, s):
                new_val = objective(c, s)
                if new_val > best_val:
                    best_val = new_val
                else:
                    c[i] = old_c
                    s[i] = old_s
            else:
                c[i] = old_c
                s[i] = old_s

        return c, s

    # Main optimization
    best_c = None
    best_s = None
    best_val = -1

    # Start with analytical construction
    centers_init = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
    ], dtype=np.float64)
    sides_init = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    c_opt, s_opt = coordinate_ascent(centers_init, sides_init, max_rounds=300)
    if is_feasible(c_opt, s_opt):
        val = objective(c_opt, s_opt)
        if val > best_val:
            best_val = val
            best_c = c_opt.copy()
            best_s = s_opt.copy()

    # Try guillotine search
    np.random.seed(42)
    g_config = guillotine_search()
    if g_config is not None:
        g_c, g_s = g_config
        if is_feasible(g_c, g_s):
            val = objective(g_c, g_s)
            if val > best_val:
                best_val = val
                best_c = g_c.copy()
                best_s = g_s.copy()
            # Refine with coordinate ascent
            c_opt, s_opt = coordinate_ascent(g_c, g_s, max_rounds=200)
            if is_feasible(c_opt, s_opt):
                val = objective(c_opt, s_opt)
                if val > best_val:
                    best_val = val
                    best_c = c_opt.copy()
                    best_s = s_opt.copy()

    # More random starts
    for start in range(30):
        if time.time() > deadline:
            break
        # Generate random feasible config
        s_rand = np.random.uniform(0.01, 0.3, size=n)
        c_rand = np.zeros((n, 2))
        feasible = True
        for i in range(n):
            placed = False
            for _ in range(200):
                cx = np.random.uniform(s_rand[i]/2, 1 - s_rand[i]/2)
                cy = np.random.uniform(s_rand[i]/2, 1 - s_rand[i]/2)
                c_rand[i] = [cx, cy]
                ok = True
                for j in range(i):
                    t = (s_rand[i] + s_rand[j]) / 2 - tol
                    if abs(cx - c_rand[j][0]) < t and abs(cy - c_rand[j][1]) < t:
                        ok = False
                        break
                if ok:
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        if not feasible or not is_feasible(c_rand, s_rand):
            continue
        c_opt, s_opt = coordinate_ascent(c_rand, s_rand, max_rounds=200)
        if is_feasible(c_opt, s_opt):
            val = objective(c_opt, s_opt)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
                best_s = s_opt.copy()
        c_ls, s_ls = local_search(c_opt, s_opt, max_iters=500)
        if is_feasible(c_ls, s_ls):
            val = objective(c_ls, s_ls)
            if val > best_val:
                best_val = val
                best_c = c_ls.copy()
                best_s = s_ls.copy()

    if best_c is None:
        best_c = centers_init.copy()
        best_s = sides_init.copy()

    if not is_feasible(best_c, best_s):
        best_c = centers_init.copy()
        best_s = sides_init.copy()

    best_s = np.maximum(best_s, 1e-9)

    return best_c, best_s


# EVOLVE-BLOCK-END
