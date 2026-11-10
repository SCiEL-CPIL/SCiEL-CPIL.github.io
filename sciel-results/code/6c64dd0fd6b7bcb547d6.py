import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # 1. Compute Forbidden Differences for mod 205
    forbidden205 = set()
    for x in range(1, m):
        forbidden205.add(pow(x, 2, m))
    forbidden205.discard(0)

    # 2. Build Adjacency Bitmasks for G_205
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden205:
                mask |= (1 << j)
        adj[i] = mask

    full_mask = (1 << m) - 1

    # Helper to extract list of nodes from mask
    def mask_to_list(mask):
        res = []
        c = mask
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            res.append(v)
            c ^= lsb
        return res

    # Helper to get compatible nodes (not in set, not adjacent to any in set)
    def get_compatible_nodes(mask):
        blocked = 0
        s = mask
        while s:
            lsb = s & (-s)
            u = lsb.bit_length() - 1
            s ^= lsb
            blocked |= adj[u]
        return full_mask & ~blocked & ~mask

    # --- CRT Structure: 205 = 5 * 41 ---
    # Map each residue x in 0..204 to (x mod 5, x mod 41)
    # Inverse: x = (a5 * 41 * inv(41,5) + a41 * 5 * inv(5,41)) % 205
    # 41 mod 5 = 1, inv(1,5)=1; 5 mod 41=5, inv(5,41)=33
    # x = (a5 * 41 + a41 * 165) % 205

    def crt_to_residue(a5, a41):
        return (a5 * 41 + a41 * 165) % 205

    def residue_to_crt(x):
        return (x % 5, x % 41)

    # Forbidden sets for components
    forb5 = set()
    for x in range(1, 5):
        forb5.add(pow(x, 2, 5))
    forb5.discard(0)  # forb5 = {1, 4}

    forb41 = set()
    for x in range(1, 41):
        forb41.add(pow(x, 2, 41))
    forb41.discard(0)

    # Two cells (i,j) and (i',j') are adjacent iff:
    # (i-i') mod 5 in forb5 AND (j-j') mod 41 in forb41

    # --- Phase 1: Find good independent sets in G_5 and G_41 ---
    # G_5: C5, alpha=2. All max independent sets: {0,2}, {0,3}, {1,3}, {1,4}, {2,4}
    # We'll use {0,2} as default but could try others.

    # G_41: Find max independent set using exact search
    m41 = 41
    adj41 = [0] * m41
    for i in range(m41):
        mask = 0
        for j in range(m41):
            if i == j: continue
            if (i - j) % m41 in forb41:
                mask |= (1 << j)
        adj41[i] = mask

    full41 = (1 << m41) - 1
    best41_mask = 0
    best41_size = 0

    # Exact branch-and-bound for G_41
    def solve_mis_41_exact():
        nonlocal best41_mask, best41_size
        start_t = time.time()

        def dfs(selected, cand):
            nonlocal best41_mask, best41_size
            if time.time() - start_t > 8:
                return

            if cand == 0:
                if selected.bit_count() > best41_size:
                    best41_size = selected.bit_count()
                    best41_mask = selected
                return

            if selected.bit_count() + cand.bit_count() <= best41_size:
                return

            # Pick vertex with min degree in cand
            min_deg = m41
            min_v = -1
            c = cand
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj41[v] & cand).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v

            if min_v == -1:
                return

            v = min_v

            # Include v
            new_cand = cand & ~adj41[v] & ~(1 << v)
            dfs(selected | (1 << v), new_cand)

            # Exclude v
            dfs(selected, cand & ~(1 << v))

        dfs(0, full41)

    solve_mis_41_exact()

    # Fallback heuristic if exact didn't finish well
    if best41_size < 10:
        for seed in range(500):
            rng = random.Random(seed)
            cand = full41
            sel = 0
            for _ in range(m41):
                if cand == 0: break
                min_deg = m41 + 1
                min_nodes = []
                c = cand
                while c:
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = (adj41[v] & cand).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_nodes = [v]
                    elif deg == min_deg:
                        min_nodes.append(v)
                if not min_nodes: break
                v = rng.choice(min_nodes)
                sel |= (1 << v)
                cand &= ~(adj41[v] | (1 << v))

            improved = True
            while improved:
                improved = False
                blocked = 0
                s = sel
                while s:
                    lsb = s & (-s)
                    u = lsb.bit_length() - 1
                    s ^= lsb
                    blocked |= adj41[u]
                comp = full41 & ~blocked & ~sel
                if comp:
                    lsb = comp & (-comp)
                    v = lsb.bit_length() - 1
                    sel |= (1 << v)
                    improved = True

            if sel.bit_count() > best41_size:
                best41_size = sel.bit_count()
                best41_mask = sel

    A41 = mask_to_list(best41_mask)

    # --- Phase 2: Build initial set via CRT product ---
    # Try all 5 max independent sets of G_5 and combine with best A41
    A5_options = [[0,2], [0,3], [1,3], [1,4], [2,4]]

    best_mask = 0
    best_size = 0

    for A5 in A5_options:
        init_mask = 0
        for a5 in A5:
            for a41 in A41:
                x = crt_to_residue(a5, a41)
                init_mask |= (1 << x)
        if init_mask.bit_count() > best_size:
            best_size = init_mask.bit_count()
            best_mask = init_mask

    # --- Phase 3: Structured Local Search exploiting CRT grid ---
    # Key insight: A set S in G_205 can be viewed as a 5x41 binary grid.
    # Cell (i,j) is in S iff the corresponding residue is in S.
    # Two cells (i,j) and (i',j') conflict iff (i-i')%5 in forb5 AND (j-j')%41 in forb41.

    # We can do local moves that:
    # 1. Add a single cell if compatible
    # 2. Swap: remove a cell, add one or more compatible cells
    # 3. Row-level move: change the entire pattern in a row

    def grid_from_mask(mask):
        """Convert bitmask to 5x41 2D array (0/1)."""
        grid = [[0]*41 for _ in range(5)]
        for i in range(5):
            for j in range(41):
                x = crt_to_residue(i, j)
                if mask & (1 << x):
                    grid[i][j] = 1
        return grid

    def mask_from_grid(grid):
        """Convert 5x41 grid to bitmask."""
        mask = 0
        for i in range(5):
            for j in range(41):
                if grid[i][j]:
                    x = crt_to_residue(i, j)
                    mask |= (1 << x)
        return mask

    def is_compatible(grid, i, j):
        """Check if adding cell (i,j) to grid keeps it independent."""
        if grid[i][j]:
            return False
        # Check against all cells in grid
        for i2 in range(5):
            for j2 in range(41):
                if grid[i2][j2]:
                    di = (i - i2) % 5
                    dj = (j - j2) % 41
                    if di in forb5 and dj in forb41:
                        return False
        return True

    def count_grid(grid):
        return sum(sum(row) for row in grid)

    def improve_grid_structured(grid, rng, max_iter=3000):
        """Local search on the 5x41 grid with structured moves."""
        current_grid = [row[:] for row in grid]
        current_size = count_grid(current_grid)

        for _ in range(max_iter):
            # Move 1: Add a compatible cell
            added = False
            for i in range(5):
                for j in range(41):
                    if is_compatible(current_grid, i, j):
                        current_grid[i][j] = 1
                        current_size += 1
                        added = True
                        break
                if added:
                    break

            if added:
                continue

            # Move 2: Single swap - remove one cell, try to add one or more
            cur_cells = [(i,j) for i in range(5) for j in range(41) if current_grid[i][j]]
            if not cur_cells:
                break

            u = rng.choice(cur_cells)
            current_grid[u[0]][u[1]] = 0
            new_size = current_size - 1

            # Try to add compatible cells to the modified grid
            improved = True
            while improved:
                improved = False
                for i in range(5):
                    for j in range(41):
                        if is_compatible(current_grid, i, j):
                            current_grid[i][j] = 1
                            new_size += 1
                            improved = True
                            break
                    if improved:
                        break

            if new_size > current_size:
                current_size = new_size
            else:
                # Revert
                current_grid[u[0]][u[1]] = 1

            # Move 3: Row pattern change - try a different independent set pattern for one row
            # This is a larger kick that can escape local optima
            if rng.random() < 0.1:
                row_idx = rng.randint(0, 4)
                # Find all valid patterns for this row given the other rows
                # A pattern is a subset of {0..40} such that for each j in pattern,
                # it's compatible with all cells in other rows

                # First, find which columns are blocked by other rows
                blocked_cols = [0]*41
                for j in range(41):
                    for i2 in range(5):
                        if i2 == row_idx:
                            continue
                        if current_grid[i2][j]:
                            di = (row_idx - i2) % 5
                            if di in forb5:
                                # j is blocked if there's a cell in row i2 at column j
                                # where (row_idx - i2) is forbidden and (j - j) = 0 is not forbidden
                                # Wait, we need to check all j2, not just j
                                pass

                # Simpler: just try to find a better pattern by random restart of this row
                # Keep other rows fixed, find max independent set in the induced subgraph
                # for this row, where a column j is "free" if it doesn't conflict with any
                # cell in other rows

                free_cols = []
                for j in range(41):
                    free = True
                    for i2 in range(5):
                        if i2 == row_idx:
                            continue
                        for j2 in range(41):
                            if current_grid[i2][j2]:
                                di = (row_idx - i2) % 5
                                dj = (j - j2) % 41
                                if di in forb5 and dj in forb41:
                                    free = False
                                    break
                        if not free:
                            break
                    if free:
                        free_cols.append(j)

                # Find max independent set in the subgraph induced by free_cols
                # where two columns j1, j2 conflict if (j1-j2) mod 41 in forb41
                # This is the same as finding max independent set in G_41 restricted to free_cols
                # For simplicity, use greedy
                if free_cols:
                    # Greedy: pick columns in order of min degree in the induced subgraph
                    remaining = set(free_cols)
                    new_pattern = set()
                    while remaining:
                        min_deg = 41
                        min_j = -1
                        for j in remaining:
                            deg = 0
                            for j2 in remaining:
                                if j2 == j:
                                    continue
                                if (j - j2) % 41 in forb41:
                                    deg += 1
                            if deg < min_deg:
                                min_deg = deg
                                min_j = j
                        if min_j == -1:
                            break
                        new_pattern.add(min_j)
                        # Remove min_j and its neighbors
                        to_remove = set()
                        for j2 in remaining:
                            if (min_j - j2) % 41 in forb41:
                                to_remove.add(j2)
                        remaining -= to_remove
                        remaining.discard(min_j)

                    # Apply new pattern to this row
                    old_count = sum(current_grid[row_idx])
                    for j in range(41):
                        current_grid[row_idx][j] = 1 if j in new_pattern else 0
                    new_count = sum(current_grid[row_idx])
                    current_size += new_count - old_count

        return current_grid, current_size

    # Run structured local search
    rng = random.Random(42)
    grid = grid_from_mask(best_mask)
    grid, best_size = improve_grid_structured(grid, rng, max_iter=5000)
    best_mask = mask_from_grid(grid)

    # --- Phase 4: Additional randomized restarts on full G_205 ---
    while time.time() < deadline - 15:
        seed = int(time.time() * 1000) % 1000000
        rng = random.Random(seed)

        # Randomized greedy start on G_205
        cand = full_mask
        sel = 0
        for _ in range(m):
            if cand == 0: break
            min_deg = m + 1
            min_nodes = []
            c = cand
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & cand).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            if not min_nodes: break
            v = rng.choice(min_nodes)
            sel |= (1 << v)
            cand &= ~(adj[v] | (1 << v))

        # Convert to grid and do structured improvement
        grid = grid_from_mask(sel)
        grid, size = improve_grid_structured(grid, rng, max_iter=2000)
        new_mask = mask_from_grid(grid)

        if size > best_size:
            best_size = size
            best_mask = new_mask

        # Also try direct bitmask local search for diversity
        def improve_mask_direct(mask, rng, max_iter=1000):
            current = mask
            current_size = current.bit_count()
            for _ in range(max_iter):
                comp = get_compatible_nodes(current)
                if comp:
                    comp_list = mask_to_list(comp)
                    v = rng.choice(comp_list)
                    current |= (1 << v)
                    current_size += 1
                    continue
                cur_list = mask_to_list(current)
                if not cur_list:
                    break
                u = rng.choice(cur_list)
                new_mask = current & ~(1 << u)
                blocked = 0
                s = new_mask
                while s:
                    lsb = s & (-s)
                    w = lsb.bit_length() - 1
                    s ^= lsb
                    blocked |= adj[w]
                valid_v = full_mask & ~blocked & ~new_mask & ~(1 << u)
                if valid_v:
                    valid_list = mask_to_list(valid_v)
                    v = rng.choice(valid_list)
                    candidate = new_mask | (1 << v)
                    improved = True
                    while improved:
                        improved = False
                        comp_c = get_compatible_nodes(candidate)
                        if comp_c:
                            comp_list = mask_to_list(comp_c)
                            v2 = rng.choice(comp_list)
                            candidate |= (1 << v2)
                            improved = True
                    if candidate.bit_count() > current_size:
                        current = candidate
                        current_size = candidate.bit_count()
            return current

        sel2 = improve_mask_direct(sel, rng, max_iter=1500)
        if sel2.bit_count() > best_size:
            best_size = sel2.bit_count()
            best_mask = sel2

    # Final polish: try to add any remaining compatible nodes
    comp = get_compatible_nodes(best_mask)
    while comp:
        lsb = comp & (-comp)
        v = lsb.bit_length() - 1
        best_mask |= (1 << v)
        comp = get_compatible_nodes(best_mask)

    # Final extraction
    res = mask_to_list(best_mask)
    return np.array(res, dtype=np.int64)
