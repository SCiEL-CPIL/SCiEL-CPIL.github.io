"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Total differences needed: n_rows * C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def scope_of(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def row_conflicts(row, used_diffs):
        """Return set of differences in this row that conflict with used_diffs."""
        conflicts = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs:
                    conflicts.add(d)
        return conflicts
    
    def greedy_row(used_diffs, k, max_val=None):
        """Build a row greedily, preferring smaller values."""
        row = [0]
        x = 1
        while len(row) < k + 1:
            if max_val is not None and x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
            x += 1
        return row
    
    def greedy_row_tight(used_diffs, k, target_scope=None):
        """Build a row with bias toward smaller values, tracking scope."""
        row = [0]
        x = 1
        while len(row) < k + 1:
            if target_scope is not None and x > target_scope:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
            x += 1
        return row
    
    def repair_row(row, used_diffs, k, max_iterations=200):
        """
        Try to reduce the scope of a row by replacing the largest entries
        with smaller values that don't create conflicts.
        Returns a new row if improvement is found, else None.
        """
        if len(row) != k + 1:
            return None
        
        current_scope = row[-1]
        improved = False
        
        for _ in range(max_iterations):
            # Find the largest entry that can potentially be reduced
            # Try to reduce from the end (largest entries first)
            best_reduction = None
            best_new_row = None
            
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try smaller values
                for new_val in range(old_val - 1, 0, -1):
                    # Check if new_val maintains strict increase
                    if pos > 0 and new_val <= row[pos - 1]:
                        continue
                    # Build candidate row
                    cand_row = row[:pos] + [new_val] + row[pos+1:]
                    # Check internal validity (strictly increasing)
                    valid_incr = all(cand_row[i] < cand_row[i+1] for i in range(len(cand_row)-1))
                    if not valid_incr:
                        continue
                    
                    # Check differences
                    cand_diffs = set()
                    conflict = False
                    for i in range(len(cand_row)):
                        for j in range(i + 1, len(cand_row)):
                            d = cand_row[j] - cand_row[i]
                            if d in used_diffs:
                                conflict = True
                                break
                            if d in cand_diffs:
                                conflict = True
                                break
                            cand_diffs.add(d)
                    if conflict:
                        continue
                    
                    # This is a valid replacement with smaller scope
                    if cand_row[-1] < current_scope or (best_reduction is None):
                        best_reduction = cand_row[-1]
                        best_new_row = cand_row
                        break  # Found a good replacement for this position
            
            if best_new_row is not None:
                row = best_new_row
                current_scope = row[-1]
                improved = True
            else:
                break
        
        return row if improved else None
    
    def try_full_repair(rows, used_diffs, k, max_rounds=50):
        """
        Iteratively repair rows to reduce scope.
        """
        rows = [list(r) for r in rows]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            progress = False
            for i in range(len(rows)):
                # Compute used diffs excluding row i
                other_diffs = set()
                for j in range(len(rows)):
                    if j == i:
                        continue
                    r = rows[j]
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            other_diffs.add(r[b] - r[a])
                
                new_row = repair_row(rows[i], other_diffs, k, max_iterations=100)
                if new_row is not None:
                    # Update used_diffs
                    # Remove old row's diffs
                    old_row = rows[i]
                    old_diffs = set()
                    for a in range(len(old_row)):
                        for b in range(a + 1, len(old_row)):
                            old_diffs.add(old_row[b] - old_row[a])
                    used_diffs -= old_diffs
                    
                    # Add new row's diffs
                    for a in range(len(new_row)):
                        for b in range(a + 1, len(new_row)):
                            used_diffs.add(new_row[b] - new_row[a])
                    
                    rows[i] = new_row
                    progress = True
            
            if not progress:
                break
        
        return rows
    
    # Phase 1: Initial greedy construction
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_diffs, k)
        if row is None:
            return None
        rows.append(row)
    
    if is_valid(rows):
        best_rows = [list(r) for r in rows]
        best_scope = scope_of(rows)
    
    # Phase 2: Try to improve by repairing
    if time.time() < deadline:
        # Recompute used_diffs for repair
        used_diffs_repair = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs_repair.add(row[j] - row[i])
        
        repaired = try_full_repair(rows, used_diffs_repair, k)
        if repaired is not None and is_valid(repaired):
            s = scope_of(repaired)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in repaired]
    
    # Phase 3: Multiple restarts with different random seeds for diversity
    if time.time() < deadline - 20:
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            random.seed(seed)
            used_diffs_r = set()
            rows_r = []
            success = True
            
            for r_idx in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    # With some probability, skip a value to create diversity
                    if random.random() < 0.1 and x + 1 < best_scope:
                        x += 1
                        continue
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs_r:
                        row.append(x)
                        used_diffs_r |= new_diffs
                    x += 1
                    if x > best_scope + 10:
                        success = False
                        break
                if not success:
                    break
                rows_r.append(row)
            
            if success and is_valid(rows_r):
                s = scope_of(rows_r)
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows_r]
                
                # Try repair on this variant
                if time.time() < deadline - 10:
                    repaired_r = try_full_repair(rows_r, used_diffs_r, k, max_rounds=30)
                    if repaired_r is not None and is_valid(repaired_r):
                        s2 = scope_of(repaired_r)
                        if s2 < best_scope:
                            best_scope = s2
                            best_rows = [list(r) for r in repaired_r]
    
    # Final fallback: if best_rows is still None, return the initial greedy
    if best_rows is None:
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used_diffs, k)
            rows.append(row)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
