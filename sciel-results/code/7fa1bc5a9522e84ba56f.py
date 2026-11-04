"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import sys

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
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed=None):
    """Greedy construction with optional randomization."""
    if seed is not None:
        random.seed(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            if randomize:
                # Try a random step within reasonable bounds
                max_step = max(1, int((k + 1) * 2.0))
                found = False
                for _attempt in range(50):
                    step = random.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback to sequential
                    x = row[-1]
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
            else:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
        rows.append(row)
    
    return rows


def _local_search_improve(rows, k, deadline, seed=None):
    """Local search: try to reduce max entry by modifying individual elements."""
    if seed is not None:
        random.seed(seed)
    
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Precompute global used diffs
    used = set()
    for r in best:
        used |= _row_diffs(r)
    
    # Helper to get diffs involving a specific position in a row
    def get_pos_diffs(row, pos):
        """Get all differences involving position pos in the row."""
        diffs = set()
        for i in range(pos):
            diffs.add(row[pos] - row[i])
        for i in range(pos + 1, len(row)):
            diffs.add(row[i] - row[pos])
        return diffs
    
    while time.time() < deadline - 2.0:
        improved = False
        current_scope = _scope(best)
        
        # Find the row with the maximum entry
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        row = best[max_row_idx]
        
        # Try to reduce the last element (which determines scope) first
        for pos in range(len(row) - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
            
            old_val = row[pos]
            lower_bound = row[pos - 1] + 1
            
            # Try values from lower_bound to old_val - 1
            for new_val in range(lower_bound, old_val):
                if time.time() > deadline - 2.0:
                    break
                
                # Calculate old diffs involving pos
                old_diffs = get_pos_diffs(row, pos)
                
                # Calculate new diffs involving pos
                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(new_val - row[i])
                for i in range(pos + 1, len(row)):
                    new_diffs.add(row[i] - new_val)
                
                # Check for conflicts
                conflict = new_diffs & (used - old_diffs)
                if conflict:
                    continue
                
                # Perform the change
                row[pos] = new_val
                used -= old_diffs
                used |= new_diffs
                
                new_scope = _scope(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
                    break
                else:
                    # Revert
                    row[pos] = old_val
                    used -= new_diffs
                    used |= old_diffs
            
            if improved:
                break
        
        if not improved:
            # Try perturbation to escape local minima
            if random.random() < 0.6:
                ridx = random.randint(0, len(best) - 1)
                r = best[ridx]
                pos = random.randint(1, len(r) - 1)
                old_val = r[pos]
                
                low = r[pos-1] + 1
                high = r[pos+1] - 1 if pos + 1 < len(r) else old_val + 20
                
                if low < high:
                    for _ in range(10):
                        new_val = random.randint(low, high)
                        if new_val == old_val:
                            continue
                        
                        old_diffs = get_pos_diffs(r, pos)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - r[i])
                        for i in range(pos + 1, len(r)):
                            new_diffs.add(r[i] - new_val)
                        
                        conflict = new_diffs & (used - old_diffs)
                        if not conflict:
                            r[pos] = new_val
                            used -= old_diffs
                            used |= new_diffs
                            break
            else:
                break
    
    return best, best_scope


def _hill_climb_search(n_rows, k, deadline, seed=None):
    """More aggressive search: multiple random restarts with hill climbing."""
    if seed is not None:
        random.seed(seed)
    
    best_rows = None
    best_scope = float('inf')
    
    # Try different strategies for initial construction
    strategies = [
        lambda: _greedy_construct(n_rows, k, randomize=False),
        lambda: _greedy_construct(n_rows, k, randomize=True),
    ]
    
    for strat in strategies:
        if time.time() > deadline - 5.0:
            break
        
        rows = strat()
        scope = _scope(rows)
        
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
        
        # Local search
        if time.time() < deadline - 3.0:
            ls_rows, ls_scope = _local_search_improve(rows, k, deadline - 3.0)
            if ls_scope < best_scope:
                best_rows = ls_rows
                best_scope = ls_scope
    
    # Multiple random restarts
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 200:
        attempts += 1
        rows = _greedy_construct(n_rows, k, randomize=True, seed=random.randint(0, 10**9))
        scope = _scope(rows)
        
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
        
        # Quick local search if promising
        if scope < best_scope + 10 and time.time() < deadline - 2.0:
            ls_rows, ls_scope = _local_search_improve(rows, k, deadline - 2.0)
            if ls_scope < best_scope:
                best_rows = ls_rows
                best_scope = ls_scope
    
    return best_rows, best_scope


def _two_phase_search(n_rows, k, deadline):
    """Two-phase: quick greedy + local search, then more intensive if time permits."""
    start_time = time.time()
    
    # Phase 1: Quick construction
    rows = _greedy_construct(n_rows, k, randomize=False)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Quick local search
    if time.time() < deadline - 10.0:
        ls_rows, ls_scope = _local_search_improve(rows, k, deadline - 10.0)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 2: More intensive search if time permits
    remaining = deadline - time.time()
    if remaining > 20.0:
        hc_rows, hc_scope = _hill_climb_search(n_rows, k, deadline, seed=int(start_time * 1000) % 100000)
        if hc_scope < best_scope:
            best_rows = hc_rows
            best_scope = hc_scope
    
    return best_rows, best_scope


def construct(n_rows, k):
    """Main entry point: construct n_rows rows of k+1 integers with distinct differences."""
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    
    # Use two-phase search
    rows, scope = _two_phase_search(n_rows, k, deadline)
    
    # Final validation
    if not _valid(rows):
        # Fallback to simple greedy
        rows = _greedy_construct(n_rows, k, randomize=False)
    
    return rows

# EVOLVE-BLOCK-END
