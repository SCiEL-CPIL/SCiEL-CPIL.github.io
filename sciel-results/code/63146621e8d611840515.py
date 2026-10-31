<<<<<<< SEARCH
def _grid_seed(w, h):
    # Place N circles in a grid inside rectangle, radius = min gap / 2
    # Choose grid cols x rows with cols*rows >= N
    best_seed = None
    best_score = -1e9
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        # cell sizes
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.999  # slight shrink for feasibility
        # build positions
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
        # quick feasibility check
        ok = True
        for i in range(N):
            ri = x[i]
            cx = x[N + i]
            cy = x[2 * N + i]
            if cx < ri - 1e-9 or cx > w - ri + 1e-9 or cy < ri - 1e-9 or cy > h - ri + 1e-9:
                ok = False
                break
        if ok:
            for i in range(N):
                for j in range(i + 1, N):
                    d2 = (x[N + i] - x[N + j]) ** 2 + (x[2 * N + i] - x[2 * N + j]) ** 2
                    if d2 < (x[i] + x[j]) ** 2 - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            score = np.sum(x[:N])
            if score > best_score:
                best_score = score
                best_seed = x.copy()
    if best_seed is None:
        # fallback: tiny circles
        best_seed = np.zeros(3 * N)
        best_seed[:N] = 1e-4
        best_seed[N:2 * N] = w / 2.0
        best_seed[2 * N:3 * N] = h / 2.0
    return best_seed
=======
def _grid_seed(w, h):
    """
    Generate multiple diverse seeds: grid, hexagonal, and randomized layouts.
    Returns list of candidate seed vectors.
    """
    seeds = []
    
    # Grid layouts
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.95  # shrink for safety
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
        if _is_feasible(x, w, tol=1e-6):
            seeds.append(x.copy())
    
    # Hexagonal packing seed
    if w > 0 and h > 0:
        x_hex = _hex_seed(w, h)
        if x_hex is not None and _is_feasible(x_hex, w, tol=1e-6):
            seeds.append(x_hex)
    
    # Randomized seeds (a few)
    rng = np.random.RandomState(42)
    for trial in range(3):
        x_rand = _random_seed(w, h, rng)
        if _is_feasible(x_rand, w, tol=1e-4):
            seeds.append(x_rand)
    
    if not seeds:
        # Fallback: tiny circles in a line
        x = np.zeros(3 * N)
        x[:N] = 1e-4
        xs = np.linspace(1e-4, w - 1e-4, N)
        x[N:2 * N] = xs
        x[2 * N:3 * N] = h / 2.0
        seeds.append(x)
    
    return seeds


def _hex_seed(w, h):
    """Generate a hexagonal packing seed."""
    # Estimate number of rows for hexagonal packing
    # Hex spacing: horizontal = r*2, vertical = r*sqrt(3)
    # Try different radii to fill the rectangle
    best = None
    best_score = -1e9
    
    for r_factor in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        r = r_factor * min(w, h)
        if r < 1e-4:
            continue
        
        # Hexagonal spacing
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        x = np.zeros(3 * N)
        idx = 0
        row = 0
        while idx < N:
            # Alternate rows offset by r
            offset = r if row % 2 == 1 else 0
            col = 0
            while idx < N:
                cx = offset + (col + 0.5) * dx
                cy = (row + 0.5) * dy
                if cx > w - r or cy > h - r:
                    break
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
                col += 1
            if idx < N:
                row += 1
        
        if idx >= N:
            if _is_feasible(x, w, tol=1e-6):
                score = np.sum(x[:N])
                if score > best_score:
                    best_score = score
                    best = x.copy()
    
    return best


def _random_seed(w, h, rng):
    """Generate a random feasible seed with small circles."""
    # Start with very small circles and try to place them
    x = np.zeros(3 * N)
    placed = 0
    
    # Try placing circles randomly
    max_attempts = 1000
    attempts = 0
    while placed < N and attempts < max_attempts:
        r = rng.uniform(0.01, 0.08)
        cx = rng.uniform(r, w - r)
        cy = rng.uniform(r, h - r)
        
        # Check if this circle fits
        ok = True
        for j in range(placed):
            dx = cx - x[N + j]
            dy = cy - x[2 * N + j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r + x[j] - 1e-8:
                ok = False
                break
        
        if ok:
            x[placed] = r
            x[N + placed] = cx
            x[2 * N + placed] = cy
            placed += 1
        attempts += 1
    
    if placed < N:
        # Fill remaining with tiny circles
        while placed < N:
            x[placed] = 1e-4
            x[N + placed] = rng.uniform(1e-4, w - 1e-4)
            x[2 * N + placed] = rng.uniform(1e-4, h - 1e-4)
            placed += 1
    
    return x
>>>>>>> REPLACE

