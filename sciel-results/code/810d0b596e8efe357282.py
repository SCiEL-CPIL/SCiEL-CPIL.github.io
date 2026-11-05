import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 105

    def greedy_build_row(used_diffs, row_idx, seed=None):
        """Build one row greedily, avoiding used differences."""
        rng = random.Random(seed) if seed is not None else random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not (diffs & used_diffs):
                row.append(x)
                used_diffs.update(diffs)
        return row

    def greedy_init(seed=None):
        rng = random.Random(seed) if seed is not None else random
        used_diffs = set()
        rows = []
        for i in range(n_rows):
            row = greedy_build_row(used_diffs, i, seed + i if seed is not None else None)
            rows.append(row)
        return rows

    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def get_M(rows):
        return max(max(r) for r in rows)

    def compute_energy(M, penalty, T):
        if penalty == 0:
            return float(M)
        lambda_pen = 100.0 * (1.0 + T / 50.0)
        return M + lambda_pen * penalty

    def rebuild_row(row_idx, rows, used_diffs_excl_row, T, rng):
        """Rebuild a single row, trying to avoid conflicts with other rows.
        Returns new row list or None if failed."""
        other_rows = [r for i, r in enumerate(rows) if i != row_idx]
        # Collect diffs from other rows
        other_diffs = set()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    other_diffs.add(r[j] - r[i])
        
        # Try multiple random seeds for the greedy build
        best_row = None
        best_score = float('inf')
        
        for attempt in range(20):
            seed = rng.randint(0, 10**9)
            row = [0]
            x = 0
            local_diffs = set()
            success = True
            while len(row) < k + 1:
                # Try several candidate values
                candidates = []
                for dx in range(1, 30):
                    cand = x + dx
                    diffs = {cand - a for a in row}
                    # Check against other rows' diffs and local diffs
                    if not (diffs & other_diffs) and not (diffs & local_diffs):
                        candidates.append(cand)
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    # Try wider search
                    for dx in range(1, 80):
                        cand = x + dx
                        diffs = {cand - a for a in row}
                        if not (diffs & other_diffs) and not (diffs & local_diffs):
                            candidates.append(cand)
                            if len(candidates) >= 3:
                                break
                
                if not candidates:
                    success = False
                    break
                
                # Pick a candidate (sometimes greedily smallest, sometimes random)
                if rng.random() < 0.6:
                    chosen = candidates[0]
                else:
                    chosen = rng.choice(candidates)
                
                row.append(chosen)
                local_diffs.update({chosen - a for a in row[:-1]})
                x = chosen
            
            if success:
                max_val = row[-1]
                if max_val < best_score:
                    best_score = max_val
                    best_row = row
        
        return best_row

    def polish_reduce_M(rows, M, deadline):
        """Try to reduce M by 1 iteratively."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            counts = get_diff_counts(rows)
            for r_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                row = rows[r_idx]
                for j_idx in range(k, 0, -1):
                    if time.time() > deadline - 2:
                        break
                    if row[j_idx] < M:
                        continue
                    old_val = row[j_idx]
                    low = row[j_idx - 1] + 1
                    high = old_val - 1
                    if high < low:
                        continue
                    for new_val in range(high, low - 1, -1):
                        valid = True
                        # Check new diffs against all other diffs (excluding this row's old diffs)
                        # First, remove this element's old diffs from consideration
                        old_diffs = set()
                        for i in range(j_idx):
                            old_diffs.add(old_val - row[i])
                        for h in range(j_idx + 1, k + 1):
                            old_diffs.add(row[h] - old_val)
                        
                        # Build set of all diffs from other rows
                        other_diffs = set()
                        for r2_idx, r2 in enumerate(rows):
                            if r2_idx == r_idx:
                                continue
                            for a in range(len(r2)):
                                for b in range(a+1, len(r2)):
                                    other_diffs.add(r2[b] - r2[a])
                        
                        # Also need to check against remaining diffs in this row (not involving j_idx)
                        same_row_other_diffs = set()
                        for i in range(len(row)):
                            for h in range(i+1, len(row)):
                                if i == j_idx or h == j_idx:
                                    continue
                                same_row_other_diffs.add(row[h] - row[i])
                        
                        # New diffs to check
                        new_diffs = set()
                        for i in range(j_idx):
                            new_diffs.add(new_val - row[i])
                        for h in range(j_idx + 1, k + 1):
                            new_diffs.add(row[h] - new_val)
                        
                        # Check no conflict
                        for d in new_diffs:
                            if d in other_diffs or d in same_row_other_diffs:
                                valid = False
                                break
                        
                        if valid:
                            row[j_idx] = new_val
                            actual_M = max(max(r) for r in rows)
                            if actual_M < M:
                                M = actual_M
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return M

    # Multi-start with row-rebuild local search
    best_rows = None
    best_M = float('inf')
    
    max_restarts = 6
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            curr_rows = greedy_init()
        else:
            if best_rows is not None and random.random() < 0.5:
                curr_rows = copy.deepcopy(best_rows)
            else:
                curr_rows = greedy_init(random.randint(0, 10**9))
        
        curr_counts = get_diff_counts(curr_rows)
        curr_penalty = get_penalty(curr_counts)
        curr_M = get_M(curr_rows)
        
        if curr_penalty == 0 and curr_M < best_M:
            best_M = curr_M
            best_rows = copy.deepcopy(curr_rows)
        
        # SA with row-rebuild moves
        T = 60.0
        cooling = 0.9997
        curr_energy = compute_energy(curr_M, curr_penalty, T)
        rng = random.Random(restart * 1000 + 42)
        
        local_deadline = start_time + (restart + 1) * time_per_restart
        
        step = 0
        while time.time() < local_deadline:
            step += 1
            
            # Choose move type: row rebuild (70%) or single element (30%)
            if rng.random() < 0.7:
                # Row rebuild move
                r_idx = rng.randint(0, n_rows - 1)
                old_row = curr_rows[r_idx][:]
                
                # Remove old row's diffs from counts
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        d = old_row[j] - old_row[i]
                        c = curr_counts[d]
                        if c > 1:
                            curr_penalty -= 1
                        curr_counts[d] -= 1
                        if curr_counts[d] == 0:
                            del curr_counts[d]
                
                # Try to rebuild
                new_row = rebuild_row(r_idx, curr_rows, None, T, rng)
                
                if new_row is None:
                    # Failed to rebuild, restore old row
                    curr_rows[r_idx] = old_row
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            d = old_row[j] - old_row[i]
                            c = curr_counts.get(d, 0)
                            if c >= 1:
                                curr_penalty += 1
                            curr_counts[d] = c + 1
                    T *= cooling
                    if T < 0.3:
                        T = 0.3
                    continue
                
                # Add new row's diffs
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        c = curr_counts.get(d, 0)
                        if c >= 1:
                            curr_penalty += 1
                        curr_counts[d] = c + 1
                
                curr_rows[r_idx] = new_row
                new_M = get_M(curr_rows)
                new_energy = compute_energy(new_M, curr_penalty, T)
                delta_E = new_energy - curr_energy
                
                if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                    curr_energy = new_energy
                    curr_M = new_M
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
                else:
                    # Revert
                    # Remove new row's diffs
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            c = curr_counts[d]
                            if c > 1:
                                curr_penalty -= 1
                            curr_counts[d] -= 1
                            if curr_counts[d] == 0:
                                del curr_counts[d]
                    
                    # Restore old row
                    curr_rows[r_idx] = old_row
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            d = old_row[j] - old_row[i]
                            c = curr_counts.get(d, 0)
                            if c >= 1:
                                curr_penalty += 1
                            curr_counts[d] = c + 1
                    
                    curr_M = get_M(curr_rows)
            else:
                # Single element move
                r_idx = rng.randint(0, n_rows - 1)
                j_idx = rng.randint(1, k)
                row = curr_rows[r_idx]
                old_val = row[j_idx]
                
                low = row[j_idx - 1] + 1
                if j_idx < k:
                    high = row[j_idx + 1] - 1
                else:
                    high = curr_M + 10
                
                if high < low:
                    T *= cooling
                    if T < 0.3:
                        T = 0.3
                    continue
                
                new_val = rng.randint(low, high)
                if new_val == old_val:
                    T *= cooling
                    if T < 0.3:
                        T = 0.3
                    continue
                
                # Remove old diffs
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                
                row[j_idx] = new_val
                
                # Add new diffs
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                
                new_M = get_M(curr_rows)
                new_energy = compute_energy(new_M, curr_penalty, T)
                delta_E = new_energy - curr_energy
                
                if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                    curr_energy = new_energy
                    curr_M = new_M
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
                else:
                    # Revert
                    row[j_idx] = old_val
                    for i in range(j_idx):
                        d = new_val - row[i]
                        c = curr_counts[d]
                        if c > 1:
                            curr_penalty -= 1
                        curr_counts[d] -= 1
                        if curr_counts[d] == 0:
                            del curr_counts[d]
                    for h in range(j_idx + 1, k + 1):
                        d = row[h] - new_val
                        c = curr_counts[d]
                        if c > 1:
                            curr_penalty -= 1
                        curr_counts[d] -= 1
                        if curr_counts[d] == 0:
                            del curr_counts[d]
                    
                    for i in range(j_idx):
                        d = old_val - row[i]
                        c = curr_counts.get(d, 0)
                        if c >= 1:
                            curr_penalty += 1
                        curr_counts[d] = c + 1
                    for h in range(j_idx + 1, k + 1):
                        d = row[h] - old_val
                        c = curr_counts.get(d, 0)
                        if c >= 1:
                            curr_penalty += 1
                        curr_counts[d] = c + 1
                    
                    curr_M = get_M(curr_rows)
            
            T *= cooling
            if T < 0.3:
                T = 0.3
    
    # Final polish
    if best_rows is not None and best_M > 0:
        best_M = polish_reduce_M(best_rows, best_M, deadline)
    
    if best_rows is None:
        best_rows = greedy_init()
    
    return best_rows
