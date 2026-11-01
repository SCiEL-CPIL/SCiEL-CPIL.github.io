import time
import random
import math

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers)
    # We want to minimize M (max entry)
    
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    def edges_from_row(row):
        """Get edge vectors (consecutive differences) from a row."""
        return [row[i+1] - row[i] for i in range(len(row)-1)]

    def row_from_edges(edges):
        """Reconstruct row from edge vectors."""
        row = [0]
        for e in edges:
            row.append(row[-1] + e)
        return row

    def all_diffs_from_edges(edges):
        """Compute all positive differences from edge vectors.
        For edges e_1, e_2, ..., e_k, the differences are all sums of consecutive edges."""
        diffs = set()
        n = len(edges)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += edges[j]
                diffs.add(s)
        return diffs

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    # ---------- Lattice-based edge vector construction ----------
    def build_row_from_edges(used_other, max_val=200, jitter=0.0, max_attempts=50):
        """Build a row using lattice-based edge vector selection.
        
        The constraint: all pairwise sums of consecutive edges (i.e., all row differences)
        must not intersect with used_other and must be unique within the row.
        
        We select edges one by one, ensuring that adding a new edge doesn't create
        any difference that conflicts with used_other or existing row differences.
        """
        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            
            edges = []
            row_diffs = set()
            success = True
            
            for pos in range(k):
                # Find valid edges at this position
                candidates = []
                for e in range(1, max_val):
                    # Compute what new differences this edge would create
                    # When we add edge e at position pos, the new differences are:
                    # e itself, and e + sum of previous consecutive edges
                    new_diffs = set()
                    # The edge itself
                    new_diffs.add(e)
                    # Sums with previous edges (going backwards)
                    s = e
                    for prev_pos in range(pos - 1, -1, -1):
                        s += edges[prev_pos]
                        new_diffs.add(s)
                    
                    # Check conflicts
                    if new_diffs & used_other:
                        continue
                    if new_diffs & row_diffs:
                        continue
                    
                    candidates.append(e)
                
                if not candidates:
                    success = False
                    break
                
                # Select edge with optional jitter
                if jitter > 0 and len(candidates) > 1:
                    # Weighted selection: prefer smaller but allow larger with probability
                    # Use exponential decay weighting
                    weights = [math.exp(-c / (20 + 5 * jitter * 10)) for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = i
                            break
                    chosen_e = candidates[chosen_idx]
                else:
                    chosen_e = candidates[0]
                
                edges.append(chosen_e)
                # Update row_diffs with all new differences
                new_diffs = set()
                new_diffs.add(chosen_e)
                s = chosen_e
                for prev_pos in range(pos - 1, -1, -1):
                    s += edges[prev_pos]
                    new_diffs.add(s)
                row_diffs |= new_diffs
            
            if success:
                row = row_from_edges(edges)
                # Verify
                if max(row) < max_val and not (all_diffs_from_edges(edges) & used_other):
                    return row, edges
        
        return None, None

    def greedy_edges(used_other, max_val=300):
        """Deterministic greedy edge-based row construction."""
        edges = []
        row_diffs = set()
        
        for pos in range(k):
            found = False
            for e in range(1, max_val):
                new_diffs = set()
                new_diffs.add(e)
                s = e
                for prev_pos in range(pos - 1, -1, -1):
                    s += edges[prev_pos]
                    new_diffs.add(s)
                
                if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                    edges.append(e)
                    row_diffs |= new_diffs
                    found = True
                    break
            
            if not found:
                return None
        
        return row_from_edges(edges)

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=150):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            
            # Alternate between edge-based and direct construction
            if attempt % 3 == 0:
                # Edge-based construction with jitter
                jitter = 0.1 + 0.1 * (attempt % 10)
                max_val = min(current_max - 1, 250)
                new_row, _ = build_row_from_edges(used_other, max_val=max_val, jitter=jitter, max_attempts=10)
            elif attempt % 3 == 1:
                # Direct greedy with jitter
                row, x = [0], 0
                current_row_diffs = set()
                success = True
                jitter = 0.15 if attempt > 15 else 0.0
                while len(row) < k + 1:
                    found = False
                    candidates = []
                    for trial_x in range(x + 1, x + 80):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            score = trial_x
                            candidates.append((score, trial_x, new))
                    
                    if not candidates:
                        success = False
                        break
                    
                    candidates.sort(key=lambda c: c[0])
                    
                    chosen_idx = 0
                    if jitter > 0 and len(candidates) > 1:
                        if random.random() < jitter:
                            top_n = min(5, len(candidates))
                            chosen_idx = random.randint(0, top_n - 1)
                    
                    _, chosen_x, chosen_new = candidates[chosen_idx]
                    row.append(chosen_x)
                    current_row_diffs |= chosen_new
                    x = row[-1]
                
                if success and max(row) < best_max:
                    test_rows = [list(r) for r in rows]
                    test_rows[row_idx] = row
                    if is_valid(test_rows):
                        best_row = list(row)
                        best_max = max(row)
            else:
                # Edge-based with more exploration
                jitter = 0.2 + 0.05 * (attempt % 8)
                max_val = min(current_max, 280)
                new_row, _ = build_row_from_edges(used_other, max_val=max_val, jitter=jitter, max_attempts=15)
            
            if new_row is not None and max(new_row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = new_row
                if is_valid(test_rows):
                    best_row = list(new_row)
                    best_max = max(new_row)
        
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Try edge-based kick first, then fall back to direct
            success = False
            for method in range(2):
                if method == 0:
                    # Edge-based with high jitter
                    for attempt in range(20):
                        jitter = 0.3 + 0.1 * random.random()
                        max_val = 350
                        new_row, _ = build_row_from_edges(used_other, max_val=max_val, jitter=jitter, max_attempts=5)
                        if new_row is not None:
                            rows[idx] = new_row
                            success = True
                            break
                else:
                    # Direct with high randomness
                    for attempt in range(25):
                        row, x = [0], 0
                        current_row_diffs = set()
                        valid_row = True
                        while len(row) < k + 1:
                            found = False
                            candidates = []
                            for trial_x in range(x + 1, x + 120):
                                new = {trial_x - a for a in row}
                                if not (new & used_other) and not (new & current_row_diffs):
                                    score = trial_x + random.random() * 20
                                    candidates.append((score, trial_x, new))
                            
                            if not candidates:
                                valid_row = False
                                break
                            
                            candidates.sort(key=lambda c: c[0])
                            
                            if len(candidates) > 1 and random.random() < 0.45:
                                top_n = min(10, len(candidates))
                                chosen_idx = random.randint(0, top_n - 1)
                            else:
                                chosen_idx = 0
                            
                            _, chosen_x, chosen_new = candidates[chosen_idx]
                            row.append(chosen_x)
                            current_row_diffs |= chosen_new
                            x = row[-1]
                        
                        if valid_row:
                            rows[idx] = row
                            success = True
                            break
                
                if success:
                    break
            
            if not success:
                # Fallback to greedy
                greedy_row = greedy_edges(used_other, max_val=400)
                if greedy_row is None:
                    greedy_row, x = [0], 0
                    row_diffs = set()
                    while len(greedy_row) < k + 1:
                        x += 1
                        new = {x - a for a in greedy_row}
                        if not (new & used_other) and not (new & row_diffs):
                            greedy_row.append(x)
                            row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        elif time_left() > 30:
            # Additional refinement: try multiple kicks and pick the best
            best_kicked = polished_kicked
            for extra_kick in range(3):
                if time_left() < 15:
                    break
                kicked2 = kick(best_rows)
                polished2 = local_polish(kicked2)
                if scope(polished2) < scope(best_kicked):
                    best_kicked = polished2
            if scope(best_kicked) < best_scope:
                best_rows = best_kicked
                best_scope = scope(best_kicked)

    # Final high-precision polish pass with tighter parameters
    if time_left() > 8:
        final_polish = local_polish(best_rows)
        if scope(final_polish) < best_scope:
            best_rows = final_polish
            best_scope = scope(final_polish)

    return best_rows
