import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_all = n_rows * total_diffs_per_row

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(r[-1] for r in best_rows)

    # CSP backtracking with forward checking to build rows within a target scope
    def csp_build_rows(target_scope, time_limit, seed=None):
        rng = random.Random(seed)
        start_time = time.time()
        used = set()
        rows = []
        # Available differences
        avail = set(range(1, target_scope + 1))
        
        def build_one_row():
            """Build one row with forward checking. Returns list or None."""
            row = [0]
            # We need to pick k more points
            # Forward check: after adding a point, ensure enough available diffs remain
            remaining_slots = k
            
            def dfs_row(current_row, remaining):
                if time.time() - start_time > time_limit:
                    return None
                if remaining == 0:
                    return list(current_row)
                
                last = current_row[-1]
                # Try candidate next values
                # Heuristic: try values that create fewer conflicts
                candidates = []
                for v in range(last + 1, target_scope + 1):
                    new_diffs = [v - a for a in current_row]
                    # Check internal distinctness
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    # Check against used
                    if any(d in used for d in new_diffs):
                        continue
                    # Forward check: after adding v, how many diffs would be used?
                    # Count how many of the new_diffs are in avail
                    # Actually we just need to ensure we can still complete
                    # Simple FC: ensure at least (remaining-1)*row_len diffs remain available after this choice
                    # This is a rough bound
                    candidates.append((v, new_diffs))
                
                # Shuffle for diversity
                rng.shuffle(candidates)
                
                for v, new_diffs in candidates:
                    # Tentatively add
                    old_used = set(used)
                    used.update(new_diffs)
                    result = dfs_row(current_row + [v], remaining - 1)
                    if result is not None:
                        return result
                    used.clear()
                    used.update(old_used)
                
                return None
            
            return dfs_row(row, k)
        
        for _ in range(n_rows):
            row = build_one_row()
            if row is None:
                return None
            # Update used
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
            rows.append(row)
        
        return rows

    deadline = time.time() + 115.0

    # Try CSP to get a tighter initial solution
    csp_start = time.time()
    csp_time = 15.0
    if time.time() + csp_time < deadline:
        # Try a few target scopes
        for target in range(best_max, max(best_max - 20, 100) - 1, -1):
            if time.time() - csp_start > csp_time:
                break
            result = csp_build_rows(target, csp_time - (time.time() - csp_start), seed=42)
            if result is not None:
                rmax = max(r[-1] for r in result)
                if rmax < best_max:
                    best_max = rmax
                    best_rows = result
                break

    # Incremental SA with diff counting
    def build_diff_counts(rows, max_scope):
        """Build a dict mapping diff value -> count of rows containing it."""
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_collisions(counts):
        """Count total collisions: sum over all d of (count[d] - 1) for count[d] > 1."""
        coll = 0
        for d, c in counts.items():
            if c > 1:
                coll += c - 1
        return coll

    def simulate_annealing(initial_rows, time_limit, initial_best_max):
        rows = [r[:] for r in initial_rows]
        current_max = max(r[-1] for r in rows)
        
        # Build diff counts
        counts = build_diff_counts(rows, initial_best_max + 50)
        current_collisions = get_collisions(counts)
        current_cost = current_collisions * 10000 + current_max
        
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        T = 50.0
        cooling = 0.9998
        start_time = time.time()
        steps = 0
        
        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_max + 30
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit
            
            if lower > upper:
                continue
            
            new_val = random.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue
            
            old_val = rows[r_idx][i_idx]
            
            # Compute delta in collisions by removing old diffs and adding new diffs
            # Remove old diffs involving position i_idx
            delta_coll = 0
            # For each other position j in this row, the diff between old_val and rows[r_idx][j]
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                if old_d != new_d:
                    # Remove old_d
                    oc = counts.get(old_d, 0)
                    if oc > 0:
                        if oc == 1:
                            del counts[old_d]
                            delta_coll -= 0  # no collision removed
                        else:
                            counts[old_d] = oc - 1
                            delta_coll -= 1  # one less collision
                    # Add new_d
                    nc = counts.get(new_d, 0)
                    if nc == 0:
                        counts[new_d] = 1
                        delta_coll += 0
                    else:
                        counts[new_d] = nc + 1
                        delta_coll += 1  # one more collision
            
            # Check if new_val is valid (strictly increasing)
            if i_idx > 0 and new_val <= rows[r_idx][i_idx - 1]:
                # Revert
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    old_d = abs(old_val - rows[r_idx][j])
                    new_d = abs(new_val - rows[r_idx][j])
                    if old_d != new_d:
                        nc = counts.get(new_d, 0)
                        if nc == 1:
                            del counts[new_d]
                        else:
                            counts[new_d] = nc - 1
                        oc = counts.get(old_d, 0)
                        if oc == 0:
                            counts[old_d] = 1
                        else:
                            counts[old_d] = oc + 1
                continue
            
            if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                # Revert
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    old_d = abs(old_val - rows[r_idx][j])
                    new_d = abs(new_val - rows[r_idx][j])
                    if old_d != new_d:
                        nc = counts.get(new_d, 0)
                        if nc == 1:
                            del counts[new_d]
                        else:
                            counts[new_d] = nc - 1
                        oc = counts.get(old_d, 0)
                        if oc == 0:
                            counts[old_d] = 1
                        else:
                            counts[old_d] = oc + 1
                continue
            
            # Apply the move
            rows[r_idx][i_idx] = new_val
            new_max = max(r[-1] for r in rows)
            new_collisions = current_collisions + delta_coll
            new_cost = new_collisions * 10000 + new_max
            
            if new_cost < current_cost:
                current_cost = new_cost
                current_collisions = new_collisions
                current_max = new_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.3:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        # Revert
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
                                nc = counts.get(new_d, 0)
                                if nc == 1:
                                    del counts[new_d]
                                else:
                                    counts[new_d] = nc - 1
                                oc = counts.get(old_d, 0)
                                if oc == 0:
                                    counts[old_d] = 1
                                else:
                                    counts[old_d] = oc + 1
                else:
                    if delta < 5000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        # Revert
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
                                nc = counts.get(new_d, 0)
                                if nc == 1:
                                    del counts[new_d]
                                else:
                                    counts[new_d] = nc - 1
                                oc = counts.get(old_d, 0)
                                if oc == 0:
                                    counts[old_d] = 1
                                else:
                                    counts[old_d] = oc + 1
            
            T *= cooling
            if T < 0.05:
                T = 0.05
            steps += 1
            
            # Periodically verify consistency
            if steps % 5000 == 0:
                actual_coll = get_collisions(counts)
                if actual_coll != current_collisions:
                    # Rebuild counts
                    counts = build_diff_counts(rows, best_max + 50)
                    current_collisions = get_collisions(counts)
                    current_max = max(r[-1] for r in rows)
                    current_cost = current_collisions * 10000 + current_max
        
        return best_local_rows, best_local_max, best_local_cost

    # Main SA loop
    sa_start = time.time()
    while time.time() < deadline - 5.0:
        # Initialize: use best_rows if valid, else random
        if best_rows is not None and all(r[0] == 0 and len(r) == row_len for r in best_rows):
            # Check if best_rows is valid
            all_diffs = []
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        all_diffs.append(r[j] - r[i])
            if len(set(all_diffs)) == len(all_diffs):
                init_rows = best_rows
            else:
                init_rows = []
                for _ in range(n_rows):
                    pts = random.sample(range(1, best_max + 1), k)
                    pts.sort()
                    init_rows.append([0] + pts)
        else:
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, best_max + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        
        remaining = deadline - time.time() - 5.0
        sa_time = min(remaining, 8.0)
        if sa_time <= 0:
            break
        
        local_rows, local_max, local_cost = simulate_annealing(init_rows, sa_time, best_max)
        
        if local_cost < 10000:  # Valid (no collisions)
            if local_max < best_max:
                best_max = local_max
                best_rows = local_rows

    return best_rows
