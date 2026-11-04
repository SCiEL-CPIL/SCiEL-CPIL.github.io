"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False):
    """Greedy construction with optional randomization."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                if row:
                    max_step = max(1, int((k + 1) * 1.5))
                    for _attempt in range(20):
                        step = random.randint(1, max_step)
                        x = row[-1] + step
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                    else:
                        x = row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                break
                            x += 1
                else:
                    x = 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _compress_row(row, used_diffs, deadline):
    """Try to compress a single row by reducing its elements while maintaining
    global validity. Returns the compressed row or the original if no improvement."""
    if time.time() > deadline:
        return list(row)
    
    n = len(row)
    # Work on a copy
    r = list(row)
    
    # Try to reduce each position from right to left
    for pos in range(n - 1, 0, -1):
        if time.time() > deadline:
            break
        
        old_val = r[pos]
        lower_bound = r[pos - 1] + 1
        
        # Try values from lower_bound to old_val - 1
        best_new = old_val
        for new_val in range(lower_bound, old_val):
            if time.time() > deadline:
                break
            
            # Compute differences involving position pos
            new_diffs = set()
            for i in range(pos):
                d = new_val - r[i]
                if d <= 0:
                    break
                new_diffs.add(d)
            
            # Check if any new diff conflicts with used_diffs (excluding diffs from this row that involve pos)
            # First, get all diffs from the current row that involve position pos
            old_row_diffs_involving_pos = set()
            for i in range(pos):
                old_row_diffs_involving_pos.add(old_val - r[i])
            
            # The "safe" used diffs are those not involving this position in the current row
            safe_used = used_diffs - old_row_diffs_involving_pos
            
            if new_diffs & safe_used:
                continue
            
            # Also check that new_diffs doesn't conflict with other diffs in this row
            # that don't involve position pos
            other_row_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if i == pos or j == pos:
                        continue
                    other_row_diffs.add(r[j] - r[i])
            
            if new_diffs & other_row_diffs:
                continue
            
            # This value works; take the smallest valid one
            best_new = new_val
            break
        
        if best_new < old_val:
            r[pos] = best_new
    
    return r


def _compress_all_rows(rows, deadline):
    """Iteratively compress all rows until no improvement or deadline."""
    rows = [list(r) for r in rows]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            
            old_row = rows[idx]
            compressed = _compress_row(old_row, used, deadline)
            
            if compressed != old_row:
                # Verify global validity
                test_rows = rows[:]
                test_rows[idx] = compressed
                if _valid(test_rows):
                    rows[idx] = compressed
                    improved = True
    
    return rows


def _search_single_row(used, k, max_val, deadline):
    """Backtrack search for a single row of length k+1 starting at 0,
    max entry <= max_val, with all differences disjoint from used.
    Uses intelligent ordering: prefer candidates that leave more future options."""
    row = [0]
    row_diffs = set()
    
    def count_future_options(x, pos):
        """Count how many valid future positions remain if we choose x at position pos."""
        # This is a heuristic: count how many values > x could potentially work
        # without immediately conflicting
        count = 0
        for future_x in range(x + 1, max_val + 1):
            new_diffs = {future_x - a for a in row}
            if not new_diffs & used and not new_diffs & row_diffs:
                count += 1
                if count > 10:  # Early exit for speed
                    break
        return count
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        
        # Generate candidate values with intelligent ordering
        candidates = []
        for x in range(row[-1] + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if new_diffs & row_diffs:
                continue
            candidates.append(x)
        
        if not candidates:
            return False
        
        # Sort by: prefer smaller x, but break ties by more future options
        # Actually, prefer x that leaves the most future options (more flexible)
        # But we also want small scope, so prefer smaller x first
        # Compromise: sort by (x, -future_options)
        scored_candidates = []
        for x in candidates:
            # Quick heuristic: smaller x is better for scope
            # But we need to ensure we can complete the row
            scored_candidates.append((x, -len(candidates)))  # Placeholder
        
        # Try candidates in order of x (smallest first)
        for x in candidates:
            new_diffs = {x - a for a in row}
            
            row.append(x)
            row_diffs.update(new_diffs)
            if backtrack(pos + 1):
                return True
            row_diffs.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _construct_global_csp(n_rows, k, deadline):
    """Global CSP approach: construct all rows simultaneously with backtracking,
    using a shared used-diff set and intelligent variable ordering."""
    used = set()
    rows = []
    
    def construct_row(r_idx, current_max):
        """Construct row r_idx with max entry <= current_max."""
        if time.time() > deadline - 5.0:
            return None
        
        row = [0]
        row_diffs = set()
        
        def backtrack(pos):
            if time.time() > deadline - 5.0:
                return False
            if pos == k + 1:
                return True
            
            # Try candidate values, preferring those that leave more options
            candidates = []
            for x in range(row[-1] + 1, current_max + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                if new_diffs & row_diffs:
                    continue
                candidates.append(x)
            
            if not candidates:
                return False
            
            # Try in order of x (smallest first for scope minimization)
            for x in candidates:
                new_diffs = {x - a for a in row}
                
                row.append(x)
                row_diffs.update(new_diffs)
                if backtrack(pos + 1):
                    return True
                row_diffs.difference_update(new_diffs)
                row.pop()
            
            return False
        
        if backtrack(1):
            return list(row)
        return None
    
    for r_idx in range(n_rows):
        if time.time() > deadline - 10.0:
            break
        
        current_max = max((max(r) for r in rows), default=0)
        found = None
        
        # Try increasing max values
        for target_max in range(current_max + 1, current_max + k + 20):
            if time.time() > deadline - 10.0:
                break
            row = construct_row(r_idx, target_max)
            if row is not None:
                found = row
                break
        
        if found is None:
            # Fallback to greedy
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            found = row
        
        rows.append(found)
        used |= _row_diffs(found)
    
    return rows


def _two_row_backtrack(used, k, max_val, deadline):
    """Try to construct two rows simultaneously, which may find better solutions
    by considering interactions between rows."""
    if time.time() > deadline:
        return None
    
    row1 = [0]
    row1_diffs = set()
    row2 = [0]
    row2_diffs = set()
    
    def backtrack(r1_pos, r2_pos):
        if time.time() > deadline:
            return False
        if r1_pos == k + 1 and r2_pos == k + 1:
            return True
        
        # Decide which row to extend next (alternate for balance)
        if r1_pos < k + 1 and (r2_pos >= k + 1 or r1_pos <= r2_pos):
            # Extend row1
            for x in range(row1[-1] + 1, max_val + 1):
                new_diffs = {x - a for a in row1}
                if new_diffs & used:
                    continue
                if new_diffs & row1_diffs:
                    continue
                if new_diffs & row2_diffs:
                    continue
                
                row1.append(x)
                row1_diffs.update(new_diffs)
                if backtrack(r1_pos + 1, r2_pos):
                    return True
                row1_diffs.difference_update(new_diffs)
                row1.pop()
        else:
            # Extend row2
            for x in range(row2[-1] + 1, max_val + 1):
                new_diffs = {x - a for a in row2}
                if new_diffs & used:
                    continue
                if new_diffs & row2_diffs:
                    continue
                if new_diffs & row1_diffs:
                    continue
                
                row2.append(x)
                row2_diffs.update(new_diffs)
                if backtrack(r1_pos, r2_pos + 1):
                    return True
                row2_diffs.difference_update(new_diffs)
                row2.pop()
        
        return False
    
    if backtrack(1, 1):
        return [list(row1), list(row2)]
    return None


def _construct_with_two_row_pairs(n_rows, k, deadline):
    """Construct rows in pairs using two-row backtracking."""
    used = set()
    rows = []
    
    # First, construct rows 0 and 1 together
    if time.time() < deadline - 30.0:
        current_max = 0
        found = None
        for target_max in range(1, 150):
            if time.time() > deadline - 30.0:
                break
            pair = _two_row_backtrack(used, k, target_max, deadline - 30.0)
            if pair is not None:
                found = pair
                break
        
        if found is not None:
            rows.extend(found)
            for r in found:
                used |= _row_diffs(r)
    
    # Construct remaining rows one by one
    while len(rows) < n_rows:
        if time.time() > deadline - 15.0:
            break
        
        current_max = max((max(r) for r in rows), default=0)
        found = None
        
        for target_max in range(current_max + 1, current_max + k + 20):
            if time.time() > deadline - 15.0:
                break
            row = _search_single_row(used, k, target_max, deadline - 15.0)
            if row is not None:
                found = row
                break
        
        if found is None:
            # Fallback to greedy
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            found = row
        
        rows.append(found)
        used |= _row_diffs(found)
    
    return rows


def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Compress the greedy solution
    if time.time() < deadline - 40.0:
        compressed = _compress_all_rows(best_rows, deadline - 40.0)
        c_scope = _scope(compressed)
        if c_scope < best_scope:
            best_rows = compressed
            best_scope = c_scope
    
    # Phase 3: Global CSP construction
    if time.time() < deadline - 30.0:
        csp_rows = _construct_global_csp(n_rows, k, deadline - 30.0)
        if csp_rows is not None and len(csp_rows) == n_rows:
            csp_scope = _scope(csp_rows)
            if csp_scope < best_scope:
                best_rows = csp_rows
                best_scope = csp_scope
                # Compress the CSP solution
                if time.time() < deadline - 20.0:
                    compressed2 = _compress_all_rows(best_rows, deadline - 20.0)
                    c2_scope = _scope(compressed2)
                    if c2_scope < best_scope:
                        best_rows = compressed2
                        best_scope = c2_scope
    
    # Phase 4: Two-row pair construction
    if time.time() < deadline - 25.0:
        pair_rows = _construct_with_two_row_pairs(n_rows, k, deadline - 25.0)
        if pair_rows is not None and len(pair_rows) == n_rows:
            pair_scope = _scope(pair_rows)
            if pair_scope < best_scope:
                best_rows = pair_rows
                best_scope = pair_scope
                # Compress
                if time.time() < deadline - 15.0:
                    compressed3 = _compress_all_rows(best_rows, deadline - 15.0)
                    c3_scope = _scope(compressed3)
                    if c3_scope < best_scope:
                        best_rows = compressed3
                        best_scope = c3_scope
    
    # Phase 5: Final compression pass
    if time.time() < deadline - 5.0:
        final_compressed = _compress_all_rows(best_rows, deadline - 3.0)
        final_scope = _scope(final_compressed)
        if final_scope < best_scope:
            best_rows = final_compressed
            best_scope = final_scope
    
    return best_rows

# EVOLVE-BLOCK-END
