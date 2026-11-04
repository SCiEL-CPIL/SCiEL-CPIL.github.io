"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 140  # 20s safety margin
    
    def row_diffs(row):
        """Get all pairwise differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def scope_of(rows):
        """Return the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    def valid(rows):
        """Check if all differences are globally distinct."""
        all_diffs = set()
        for row in rows:
            rd = row_diffs(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True
    
    def greedy_row(used, k, max_val, randomize=False):
        """Build one row greedily, optionally with randomization."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            candidates = []
            for dx in range(1, max_val - x + 1):
                new_x = x + dx
                if new_x > max_val:
                    break
                new_diffs = {new_x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(dx)
            if not candidates:
                return None
            if randomize and candidates:
                # Pick randomly from the first several candidates for diversity
                pick_idx = random.randint(0, min(len(candidates) - 1, 5))
                dx = candidates[pick_idx]
            else:
                dx = candidates[0]
            x += dx
            row.append(x)
            used |= {x - a for a in row[:-1]}
        return row
    
    def greedy_all(n_rows, k, max_val, randomize=False, seed=None):
        """Build all rows greedily."""
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, max_val, randomize)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    def local_search(rows, k, max_val, time_limit):
        """Try to reduce scope by moving entries."""
        rows = [list(r) for r in rows]
        best_scope = scope_of(rows)
        start = time.time()
        improved = True
        
        while improved and time.time() - start < time_limit:
            improved = False
            # Try to reduce each entry in each row
            for ri in range(len(rows)):
                for ei in range(len(rows[ri]) - 1, 0, -1):  # from last to first
                    old_val = rows[ri][ei]
                    # Try to reduce this entry
                    for new_val in range(old_val - 1, rows[ri][ei-1], -1):
                        rows[ri][ei] = new_val
                        # Check validity
                        if valid(rows):
                            new_scope = scope_of(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                improved = True
                                break
                            elif new_scope == best_scope:
                                # Keep it if same scope but might help future moves
                                # Actually, let's only keep if strictly better scope
                                rows[ri][ei] = old_val
                            else:
                                rows[ri][ei] = old_val
                        else:
                            rows[ri][ei] = old_val
                    if time.time() - start > time_limit:
                        break
                if time.time() - start > time_limit:
                    break
            if time.time() - start > time_limit:
                break
        
        return rows, best_scope
    
    # First, find a valid solution with greedy
    # Start with a reasonable upper bound
    # Simple counting: need 144 distinct differences, so max entry >= 144
    # But we need to account for the structure
    
    best_rows = None
    best_scope = float('inf')
    
    # Try pure greedy first
    for max_val in range(112, 200):
        rows = greedy_all(n_rows, k, max_val, randomize=False)
        if rows is not None and valid(rows):
            s = scope_of(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            break
    
    if best_rows is None:
        # Shouldn't happen, but fallback
        rows = greedy_all(n_rows, k, 500, randomize=False)
        best_rows = rows
        best_scope = scope_of(rows)
    
    # Now do local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Use some time for local search
        search_time = min(remaining_time - 2, 60)
        best_rows, best_scope = local_search(best_rows, k, best_scope, search_time)
    
    # Try multiple random restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        attempts = 0
        while time.time() < deadline - 5:
            attempts += 1
            seed = random.randint(0, 1000000)
            rows = greedy_all(n_rows, k, best_scope + 50, randomize=True, seed=seed)
            if rows is not None and valid(rows):
                s = scope_of(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                    # Do local search on this
                    remaining = deadline - time.time()
                    if remaining > 5:
                        search_time = min(remaining - 2, 30)
                        best_rows, best_scope = local_search(best_rows, k, best_scope, search_time)
    
    return best_rows
# EVOLVE-BLOCK-END
