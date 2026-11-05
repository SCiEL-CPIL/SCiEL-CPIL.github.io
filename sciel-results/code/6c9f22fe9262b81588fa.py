import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s, task says 120s. Use 110s to be safe.
    deadline = start_time + 110
    
    # Gap-vector representation: each row stored as gaps [g1, g2, ..., gk]
    # where row[i] = sum(gaps[0:i]) for i >= 1, row[0] = 0
    
    def gaps_to_row(gaps):
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        return row
    
    def row_to_gaps(row):
        return [row[i+1] - row[i] for i in range(len(row)-1)]
    
    def compute_all_diffs(rows):
        """Return dict: diff -> count, and list of (row_idx, i, j) for each diff occurrence"""
        diff_counts = {}
        diff_occurrences = []
        for r_idx, r in enumerate(rows):
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    diff_occurrences.append((r_idx, i, j, d))
        return diff_counts, diff_occurrences
    
    def get_penalty(diff_counts):
        pen = 0
        for c in diff_counts.values():
            if c > 1:
                pen += c - 1
        return pen
    
    def get_M(rows):
        return max(max(r) for r in rows)
    
    # Multi-start with different random seeds for greedy init
    def greedy_init(rng):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows
    
    def random_init(rng, max_scope):
        """Generate random rows with entries up to max_scope"""
        rows = []
        for _ in range(n_rows):
            # Generate k random increasing values
            candidates = list(range(1, max_scope + 1))
            rng.shuffle(candidates)
            selected = sorted(candidates[:k])
            row = [0] + selected
            rows.append(row)
        return rows
    
    best_rows = None
    best_M = float('inf')
    
    # Phase 1: Multi-start local search
    n_starts = 5
    time_per_start = (deadline - start_time) / n_starts
    
    for start_i in range(n_starts):
        if time.time() >= deadline - 5:
            break
            
        rng = random.Random(start_i * 1000 + 42)
        
        # Alternate between greedy and random init
        if start_i % 2 == 0:
            curr_rows = greedy_init(rng)
        else:
            # Random init with scope near best known
            scope_for_init = best_M if best_M < float('inf') else 200
            curr_rows = random_init(rng, scope_for_init)
        
        # Compute initial state
        curr_diff_counts, _ = compute_all_diffs(curr_rows)
        curr_penalty = get_penalty(curr_diff_counts)
        curr_M = get_M(curr_rows)
        
        if curr_penalty == 0 and curr_M < best_M:
            best_M = curr_M
            best_rows = copy.deepcopy(curr_rows)
        
        # SA on gap vectors
        curr_gaps = [row_to_gaps(r) for r in curr_rows]
        
        # Energy: M + lambda * penalty + mu * (max_scope_exceedance)
        # Allow navigating through invalid states by using penalty rather than hard constraint
        T = max(curr_M * 0.5, 20.0)
        cooling = 0.9995
        lambda_pen = 50.0
        mu_scope = 100.0  # Hard penalty for exceeding current best scope
        
        curr_energy = curr_M + lambda_pen * curr_penalty
        if curr_M > best_M:
            curr_energy += mu_scope * (curr_M - best_M)
        
        local_best_M = curr_M if curr_penalty == 0 else float('inf')
        local_best_rows = copy.deepcopy(curr_rows) if curr_penalty == 0 else None
        
        # SA loop with gap-vector moves
        while time.time() < deadline - 10:
            # Generate move: change one gap in one row
            r_idx = rng.randint(0, n_rows - 1)
            gaps = curr_gaps[r_idx]
            g_idx = rng.randint(0, k - 1)
            
            old_gap = gaps[g_idx]
            
            # Determine valid range for new gap
            # Minimum gap is 1
            # Maximum: ensure the resulting entry doesn't exceed some bound
            # Compute current row to check bounds
            row = gaps_to_row(gaps)
            
            # The new value at position g_idx+1 changes
            # New value = old_value + delta where delta = new_gap - old_gap
            # We need all entries to remain positive and increasing
            
            # Simple approach: try gap values in a range around current
            max_gap = max(old_gap * 2, 10)
            min_gap = 1
            
            new_gap = rng.randint(min_gap, max_gap)
            
            if new_gap == old_gap:
                continue
            
            # Apply change
            gaps[g_idx] = new_gap
            new_row = gaps_to_row(gaps)
            
            # Check validity: strictly increasing starting at 0
            valid = True
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i-1]:
                    valid = False
                    break
            
            if not valid:
                gaps[g_idx] = old_gap
                continue
            
            # Compute new M
            new_M = max(new_row[-1], max(max(r) for r in curr_rows if r is not curr_rows[r_idx]))
            # More efficient: track max per row
            other_max = 0
            for i, r in enumerate(curr_rows):
                if i != r_idx:
                    if max(r) > other_max:
                        other_max = max(r)
            new_M = max(new_row[-1], other_max)
            
            # Compute difference changes incrementally
            # Old row diffs that are removed
            old_row = gaps_to_row([g if i != g_idx else old_gap for i, g in enumerate(gaps)])
            
            # Actually, let's just recompute for the changed row
            # Remove old diffs for this row
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    d = old_row[j] - old_row[i]
                    c = curr_diff_counts.get(d, 0)
                    if c > 1:
                        curr_penalty -= 1
                    curr_diff_counts[d] = c - 1
                    if curr_diff_counts[d] == 0:
                        del curr_diff_counts[d]
            
            # Add new diffs for this row
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    c = curr_diff_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_diff_counts[d] = c + 1
            
            # Update row in curr_rows
            curr_rows[r_idx] = new_row
            
            new_energy = new_M + lambda_pen * curr_penalty
            if new_M > best_M:
                new_energy += mu_scope * (new_M - best_M)
            
            delta_E = new_energy - curr_energy
            
            # Acceptance
            accept = False
            if delta_E < 0:
                accept = True
            else:
                if T > 0.01:
                    ratio = math.exp(-delta_E / T)
                    if rng.random() < ratio:
                        accept = True
            
            if accept:
                curr_energy = new_energy
                curr_M = new_M
                
                if curr_penalty == 0:
                    if curr_M < local_best_M:
                        local_best_M = curr_M
                        local_best_rows = copy.deepcopy(curr_rows)
                        if curr_M < best_M:
                            best_M = curr_M
                            best_rows = copy.deepcopy(curr_rows)
            else:
                # Revert
                gaps[g_idx] = old_gap
                curr_rows[r_idx] = old_row
                
                # Revert diff counts
                # Remove new diffs
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        c = curr_diff_counts[d]
                        if c > 1:
                            curr_penalty -= 1
                        curr_diff_counts[d] = c - 1
                        if curr_diff_counts[d] == 0:
                            del curr_diff_counts[d]
                
                # Add old diffs
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        d = old_row[j] - old_row[i]
                        c = curr_diff_counts.get(d, 0)
                        if c >= 1:
                            curr_penalty += 1
                        curr_diff_counts[d] = c + 1
                
                # Recompute M
                curr_M = max(max(r) for r in curr_rows)
                curr_energy = curr_M + lambda_pen * curr_penalty
                if curr_M > best_M:
                    curr_energy += mu_scope * (curr_M - best_M)
            
            T *= cooling
            if T < 0.5:
                T = 0.5
        
        # After SA, try to polish
        if local_best_rows is not None:
            if local_best_M < best_M:
                best_M = local_best_M
                best_rows = copy.deepcopy(local_best_rows)
    
    # Phase 2: Final polish - try to reduce M by moving individual entries
    if best_rows is not None and time.time() < deadline - 3:
        polish_deadline = deadline - 2
        curr_rows = copy.deepcopy(best_rows)
        curr_M = get_M(curr_rows)
        
        # Try to reduce each entry slightly
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            curr_diff_counts, _ = compute_all_diffs(curr_rows)
            curr_penalty = get_penalty(curr_diff_counts)
            
            if curr_penalty != 0:
                break
            
            for r_idx in range(n_rows):
                if time.time() >= polish_deadline:
                    break
                row = curr_rows[r_idx]
                for j_idx in range(k, 0, -1):  # Try to reduce from the end
                    if time.time() >= polish_deadline:
                        break
                    old_val = row[j_idx]
                    # Try to reduce this value
                    new_low = row[j_idx - 1] + 1
                    new_high = old_val - 1
                    if new_high < new_low:
                        continue
                    
                    # Try values from new_high down to new_low
                    for candidate in range(new_high, new_low - 1, -1):
                        # Check if this creates any duplicate diffs
                        new_diffs = set()
                        for i in range(j_idx):
                            new_diffs.add(candidate - row[i])
                        for h in range(j_idx + 1, k + 1):
                            new_diffs.add(row[h] - candidate)
                        
                        # Check against all other diffs
                        conflict = False
                        for d in new_diffs:
                            # Count how many times d appears in current state
                            # excluding the diffs we're about to remove
                            pass
                        
                        # Simpler: temporarily set and check
                        row[j_idx] = candidate
                        # Check all diffs in this row
                        row_diffs = {}
                        for i in range(len(row)):
                            for h in range(i+1, len(row)):
                                d = row[h] - row[i]
                                row_diffs[d] = row_diffs.get(d, 0) + 1
                        
                        # Check for conflicts within row or with other rows
                        conflict = False
                        for d, cnt in row_diffs.items():
                            if cnt > 1:
                                conflict = True
                                break
                            # Check if d appears in other rows
                            if not conflict:
                                for other_r_idx in range(n_rows):
                                    if other_r_idx == r_idx:
                                        continue
                                    other_row = curr_rows[other_r_idx]
                                    for i in range(len(other_row)):
                                        for h in range(i+1, len(other_row)):
                                            if other_row[h] - other_row[i] == d:
                                                conflict = True
                                                break
                                        if conflict:
                                            break
                                    if conflict:
                                        break
                        
                        if not conflict:
                            # Accept this reduction
                            new_M = max(max(r) for r in curr_rows)
                            if new_M < curr_M:
                                curr_M = new_M
                                improved = True
                                break
                        else:
                            row[j_idx] = old_val
                            break
                    
                    if time.time() >= polish_deadline:
                        break
            
            if curr_M < best_M:
                best_M = curr_M
                best_rows = copy.deepcopy(curr_rows)
    
    if best_rows is None:
        # Fallback: return greedy result
        best_rows = greedy_init(random.Random(42))
    
    return best_rows
