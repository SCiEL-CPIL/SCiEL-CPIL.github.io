"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 142.0

    def max_entry(rows):
        return max(max(row) for row in rows)

    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def compute_violations_and_max(rows):
        """Return (violations, max_entry) with O(n*k^2) computation."""
        diff_count = {}
        m = 0
        for row in rows:
            for val in row:
                if val > m:
                    m = val
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        violations = sum(c - 1 for c in diff_count.values() if c > 1)
        return violations, m

    # --- Phase 1: Simultaneous greedy construction ---
    # Build all rows position-by-position (not row-by-row) to balance conflict
    def simultaneous_greedy(scope_hint=None):
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        
        # Build position by position across all rows
        for pos in range(1, k + 1):
            # For each position, assign values to all rows
            # Try to spread values to minimize conflicts
            candidates_by_row = {}
            for r in range(n_rows):
                prev_val = rows[r][pos - 1]
                max_val = scope_hint if scope_hint else 500
                cands = []
                for x in range(prev_val + 1, max_val + 1):
                    # Check new diffs against used_diffs
                    conflict = False
                    new_diffs = set()
                    for i in range(pos):
                        d = x - rows[r][i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        cands.append(x)
                candidates_by_row[r] = cands
            
            # If any row has no candidates, we're stuck
            if any(len(c) == 0 for c in candidates_by_row.values()):
                return None
            
            # Assign values: try to use different values across rows
            # Sort rows by fewest candidates first (most constrained)
            row_order = sorted(range(n_rows), key=lambda r: len(candidates_by_row[r]))
            
            assigned = {}
            for r in row_order:
                # Pick the smallest candidate that doesn't conflict with already-assigned
                # in this position (to keep values spread out)
                found = False
                for x in candidates_by_row[r]:
                    # Check if x creates conflicts with already-assigned values in this position
                    # (differences between different rows at same position don't matter,
                    #  only within-row diffs matter, so we just need x not to create
                    #  new used_diffs conflicts)
                    # Actually, since we're building position by position, the new diffs
                    # from x are x - rows[r][i] for i < pos, which we already checked.
                    # We just need to avoid reusing the same x across rows if possible
                    # to keep the structure diverse.
                    if x not in assigned.values():
                        rows[r].append(x)
                        assigned[r] = x
                        for i in range(pos):
                            used_diffs.add(x - rows[r][i])
                        found = True
                        break
                if not found:
                    # Allow reuse if necessary
                    x = candidates_by_row[r][0]
                    rows[r].append(x)
                    assigned[r] = x
                    for i in range(pos):
                        used_diffs.add(x - rows[r][i])
        
        return rows

    best_rows = simultaneous_greedy()
    if best_rows is None:
        # Fallback: row-by-row greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    best_max = max_entry(best_rows)
    best_violations, _ = compute_violations_and_max(best_rows)

    # --- Phase 2: Incremental SA with conflict-aware moves ---
    current_rows = [r[:] for r in best_rows]
    current_violations, current_max = compute_violations_and_max(current_rows)
    
    # Track diff counts incrementally
    diff_counts = defaultdict(int)
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] += 1
    
    def get_violations_from_counts(counts):
        return sum(c - 1 for c in counts.values() if c > 1)
    
    def try_move(r, pos, new_val):
        """Try changing rows[r][pos] to new_val. Returns (new_violations, new_max, old_diffs) or None if invalid."""
        row = current_rows[r]
        old_val = row[pos]
        if new_val == old_val:
            return None
        if new_val <= row[pos - 1] or (pos < k and new_val >= row[pos + 1]):
            return None
        
        # Collect old diffs involving position pos
        old_diffs = []
        for i in range(len(row)):
            if i != pos:
                d = abs(new_val - row[i]) if False else (row[i] - new_val if i < pos else new_val - row[i])
                # Actually: d = row[j] - row[i] where j > i
                # If i < pos: d = old_val - row[i]
                # If i > pos: d = row[i] - old_val
                if i < pos:
                    d = old_val - row[i]
                else:
                    d = row[i] - old_val
                old_diffs.append(d)
        
        # Remove old diffs from counts
        for d in old_diffs:
            diff_counts[d] -= 1
        
        # Compute new diffs
        new_diffs = []
        valid = True
        for i in range(len(row)):
            if i != pos:
                if i < pos:
                    d = new_val - row[i]
                else:
                    d = row[i] - new_val
                if d <= 0:
                    valid = False
                    break
                new_diffs.append(d)
        
        if not valid:
            # Restore
            for d in old_diffs:
                diff_counts[d] += 1
            return None
        
        # Check if new diffs create violations
        # Temporarily add new diffs
        for d in new_diffs:
            diff_counts[d] += 1
        
        new_violations = get_violations_from_counts(diff_counts)
        new_max = max(max(row2) for row2 in current_rows)
        
        return new_violations, new_max, old_diffs, new_diffs

    def undo_move(r, pos, old_diffs, new_diffs):
        """Undo a move by removing new_diffs and restoring old_diffs."""
        for d in new_diffs:
            diff_counts[d] -= 1
        for d in old_diffs:
            diff_counts[d] += 1

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_max + 20
        
        if lower > upper:
            T *= cooling
            continue
        
        # Bias toward smaller values to reduce max
        if random.random() < 0.5:
            span = upper - lower + 1
            new_val = lower + int(random.random() * min(span * 0.3, span))
        else:
            new_val = random.randint(lower, upper)
        
        result = try_move(r, i, new_val)
        if result is None:
            T *= cooling
            continue
        
        new_violations, new_max, old_diffs, new_diffs = result
        current_rows[r][i] = new_val
        
        # Cost: prioritize reducing violations, then max
        old_cost = current_violations * 1000 + current_max
        new_cost = new_violations * 1000 + new_max
        
        delta = new_cost - old_cost
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_violations = new_violations
            current_max = new_max
            if current_violations == 0 and current_max < best_max:
                best_max = current_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            undo_move(r, i, old_diffs, new_diffs)
            current_rows[r][i] = old_val
        
        T *= cooling

    # --- Phase 3: Conflict-driven repair ---
    # If we still have violations, try to resolve them specifically
    if best_violations > 0 or current_violations > 0:
        repair_rows = [r[:] for r in current_rows]
        repair_diff_counts = defaultdict(int)
        for row in repair_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    repair_diff_counts[d] += 1
        
        repair_violations = get_violations_from_counts(repair_diff_counts)
        repair_max = max(max(row) for row in repair_rows)
        
        T_rep = 10.0
        for _ in range(5000):
            if time.time() > deadline:
                break
            if repair_violations == 0:
                break
            
            # Find a violated diff
            violated_diffs = [d for d, c in repair_diff_counts.items() if c > 1]
            if not violated_diffs:
                break
            target_diff = random.choice(violated_diffs)
            
            # Find rows that contain this diff
            affected = []
            for r, row in enumerate(repair_rows):
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if row[j] - row[i] == target_diff:
                            affected.append((r, i, j))
            
            if not affected:
                break
            
            # Pick one affected position to modify
            r, i, j = random.choice(affected)
            # Modify position i or j (whichever has more freedom)
            if random.random() < 0.5:
                pos = i
            else:
                pos = j
            
            row = repair_rows[r]
            old_val = row[pos]
            lower = row[pos - 1] + 1 if pos > 0 else 1
            upper = row[pos + 1] - 1 if pos < k else repair_max + 10
            
            if lower > upper:
                continue
            
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            # Compute old diffs
            old_diffs = []
            for idx in range(len(row)):
                if idx != pos:
                    if idx < pos:
                        d = old_val - row[idx]
                    else:
                        d = row[idx] - old_val
                    old_diffs.append(d)
            
            # Remove old diffs
            for d in old_diffs:
                repair_diff_counts[d] -= 1
            
            # Check new diffs
            new_diffs = []
            valid = True
            for idx in range(len(row)):
                if idx != pos:
                    if idx < pos:
                        d = new_val - row[idx]
                    else:
                        d = row[idx] - new_val
                    if d <= 0:
                        valid = False
                        break
                    new_diffs.append(d)
            
            if not valid:
                for d in old_diffs:
                    repair_diff_counts[d] += 1
                continue
            
            for d in new_diffs:
                repair_diff_counts[d] += 1
            
            new_rep_violations = get_violations_from_counts(repair_diff_counts)
            new_rep_max = max(max(row2) for row2 in repair_rows)
            
            old_cost = repair_violations * 1000 + repair_max
            new_cost = new_rep_violations * 1000 + new_rep_max
            
            delta = new_cost - old_cost
            if delta <= 0 or random.random() < math.exp(-delta / T_rep):
                repair_rows[r][pos] = new_val
                repair_violations = new_rep_violations
                repair_max = new_rep_max
                if repair_violations == 0 and repair_max < best_max:
                    best_max = repair_max
                    best_rows = [rr[:] for rr in repair_rows]
            else:
                for d in new_diffs:
                    repair_diff_counts[d] -= 1
                for d in old_diffs:
                    repair_diff_counts[d] += 1
            
            T_rep *= 0.999

    # Final validation
    final_diffs = all_diffs(best_rows)
    if len(final_diffs) != len(set(final_diffs)):
        # Invalid, try to fix or return best effort
        pass

    return best_rows
# EVOLVE-BLOCK-END
