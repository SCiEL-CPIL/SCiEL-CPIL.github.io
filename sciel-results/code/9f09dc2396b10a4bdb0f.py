"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import itertools

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


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization and seed offset."""
    rng = random.Random(seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.0))
                found = False
                for _attempt in range(30):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
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


def _compress_rows(rows, k, deadline):
    """Aggressively compress rows by trying to reduce each element.
    Uses incremental difference tracking for speed."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Build global used set
    used = set()
    for r in best:
        used |= _row_diffs(r)
    
    n_rows = len(best)
    k1 = k + 1
    
    while time.time() < deadline:
        improved = False
        
        # Sort positions by current value descending to target largest first
        positions = []
        for ri in range(n_rows):
            for pi in range(1, k1):
                positions.append((best[ri][pi], ri, pi))
        positions.sort(reverse=True)
        
        for val, ri, pi in positions:
            if time.time() > deadline:
                break
            
            row = best[ri]
            old_val = row[pi]
            if old_val <= 1:
                continue
            
            lower = row[pi - 1] + 1
            upper = old_val - 1
            
            if lower > upper:
                continue
            
            # Try to find a smaller value that works
            # First try the smallest possible, then scan upward
            found_new = None
            
            # Try from lower bound upward
            for new_val in range(lower, upper + 1):
                # Compute new diffs for this position
                new_diffs = set()
                for i in range(pi):
                    new_diffs.add(new_val - row[i])
                for i in range(pi + 1, k1):
                    new_diffs.add(row[i] - new_val)
                
                # Old diffs for this position
                old_diffs = set()
                for i in range(pi):
                    old_diffs.add(old_val - row[i])
                for i in range(pi + 1, k1):
                    old_diffs.add(row[i] - old_val)
                
                # Check: new_diffs must not conflict with (used - old_diffs)
                # i.e., new_diffs & (used - old_diffs) must be empty
                # And new_diffs must not have internal duplicates (guaranteed since row is strict)
                remaining = used - old_diffs
                if new_diffs & remaining:
                    continue
                
                found_new = new_val
                break
            
            if found_new is not None:
                # Apply the change
                row[pi] = found_new
                
                # Update used set
                old_diffs = set()
                for i in range(pi):
                    old_diffs.add(old_val - row[i])
                for i in range(pi + 1, k1):
                    old_diffs.add(row[i] - old_val)
                
                new_diffs = set()
                for i in range(pi):
                    new_diffs.add(found_new - row[i])
                for i in range(pi + 1, k1):
                    new_diffs.add(row[i] - found_new)
                
                used -= old_diffs
                used |= new_diffs
                
                new_scope = _scope(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
                # Even if scope didn't decrease, the row is now more compressed,
                # which may enable further reductions elsewhere
                improved = True
        
        if not improved:
            # Try a perturbation to escape local optimum
            if random.random() < 0.3:
                ri = random.randint(0, n_rows - 1)
                pi = random.randint(1, k1 - 2)
                saved = best[ri][pi]
                # Try a random increase (perturbation)
                delta = random.randint(1, 5)
                best[ri][pi] = saved + delta
                if _valid(best):
                    used = set()
                    for r in best:
                        used |= _row_diffs(r)
                else:
                    best[ri][pi] = saved
            else:
                break
    
    return best, best_scope


def _two_row_backtrack(used, k, max_val, deadline):
    """Try to find two rows of length k+1 simultaneously with backtracking.
    This allows better coordination between rows."""
    row1 = [0]
    row2 = [0]
    rd1 = set()
    rd2 = set()
    
    def bt1(pos1, pos2):
        if time.time() > deadline:
            return False
        if pos1 == k + 1 and pos2 == k + 1:
            return True
        if pos1 == k + 1:
            # Continue with row2
            return bt2(pos2)
        
        # Try candidates for row1
        start1 = row1[-1] + 1
        for x1 in range(start1, max_val + 1):
            nd1 = set()
            for a in row1:
                nd1.add(x1 - a)
            if nd1 & used or nd1 & rd1:
                continue
            
            row1.append(x1)
            rd1.update(nd1)
            
            if bt1(pos1 + 1, pos2):
                return True
            
            rd1.difference_update(nd1)
            row1.pop()
        
        return False
    
    def bt2(pos2):
        if time.time() > deadline:
            return False
        if pos2 == k + 1:
            return True
        
        start2 = row2[-1] + 1
        for x2 in range(start2, max_val + 1):
            nd2 = set()
            for a in row2:
                nd2.add(x2 - a)
            if nd2 & used or nd2 & rd2:
                continue
            if nd2 & rd1:
                continue
            
            row2.append(x2)
            rd2.update(nd2)
            
            if bt2(pos2 + 1):
                return True
            
            rd2.difference_update(nd2)
            row2.pop()
        
        return False
    
    if bt1(1, 1):
        return list(row1), list(row2)
    return None, None


def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for seed in range(20):
        if time.time() > deadline - 20.0:
            break
        rows = _greedy_construct(n_rows, k, randomize=(seed > 0), seed_offset=seed * 1000 + 42)
        sc = _scope(rows)
        if sc < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = sc
    
    # Phase 2: Aggressive compression
    if time.time() < deadline - 10.0:
        comp_rows, comp_scope = _compress_rows(best_rows, k, deadline - 10.0)
        if comp_scope < best_scope:
            best_rows = comp_rows
            best_scope = comp_scope
    
    # Phase 3: More random restarts + compression
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 80:
        attempts += 1
        seed = attempts * 7919 + 13
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed)
        sc = _scope(rows)
        
        # Quick compress even if not better than best yet, to see if it can be improved
        if time.time() < deadline - 15.0:
            comp_rows, comp_scope = _compress_rows(rows, k, deadline - 15.0)
            if comp_scope < best_scope:
                best_rows = comp_rows
                best_scope = comp_scope
        elif sc < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = sc
    
    # Phase 4: Final compression pass
    if time.time() < deadline - 3.0:
        comp_rows, comp_scope = _compress_rows(best_rows, k, deadline - 1.0)
        if comp_scope < best_scope:
            best_rows = comp_rows
            best_scope = comp_scope
    
    return best_rows

# EVOLVE-BLOCK-END
