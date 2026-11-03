import random
import time
import math

_DEADLINE = 145.0  # Safety margin under 150s wall-clock

def _row_diffs(row):
    """Compute set of positive differences for a single row."""
    n = len(row)
    s = set()
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _validate(rows):
    """Check if the solution is valid."""
    used = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy(n_rows, k, rng):
    """
    Greedy construction with random tie-breaking.
    Tries to place elements as small as possible.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        # We want to choose the next element x such that x - a for all a in row are not in used.
        # To minimize scope, we pick the smallest such x.
        # However, to allow for better packing later, we can add some randomness.
        # For now, deterministic greedy is a safe baseline.
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                x += 1
            else:
                x += 1
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Simulated Annealing with suffix-shift moves.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute diffs
    row_diffs = [_row_diffs(r) for r in rows]
    used = set()
    for s in row_diffs:
        used |= s
        
    temperature = best_scope * 0.5
    cooling = 0.9995
    min_temp = 0.1
    
    def try_shift(r, j):
        """
        Try to shift rows[r][j:] to the left by 1.
        Returns True if valid and applied.
        """
        nonlocal used
        row = rows[r]
        if j <= 0 or j >= len(row):
            return False
        if row[j] - 1 <= row[j-1]:
            return False
            
        # Old diffs to remove
        # Diffs within the suffix
        old_suffix = row[j:]
        old_suffix_diffs = set()
        for i in range(len(old_suffix)):
            ai = old_suffix[i]
            for l in range(i+1, len(old_suffix)):
                old_suffix_diffs.add(old_suffix[l] - ai)
                
        # Diffs between prefix and suffix
        prefix = row[:j]
        cross_diffs = set()
        for a in prefix:
            for b in old_suffix:
                cross_diffs.add(b - a)
                
        old_diffs = old_suffix_diffs | cross_diffs
        
        # New suffix
        new_suffix = [x - 1 for x in old_suffix]
        
        # New diffs
        new_suffix_diffs = set()
        for i in range(len(new_suffix)):
            ai = new_suffix[i]
            for l in range(i+1, len(new_suffix)):
                new_suffix_diffs.add(new_suffix[l] - ai)
                
        new_cross_diffs = set()
        for a in prefix:
            for b in new_suffix:
                new_cross_diffs.add(b - a)
                
        new_diffs = new_suffix_diffs | new_cross_diffs
        
        # Check collisions with other rows' diffs
        others = used - old_diffs
        if new_diffs & others:
            return False
            
        # Check internal collisions in new_diffs (should be none if row is strictly increasing, but let's be safe)
        # Actually, new_suffix is strictly increasing, so new_suffix_diffs are distinct.
        # Cross diffs are distinct from each other?
        # If b1 - a1 == b2 - a2, it's a collision within the new row.
        # We must ensure new_diffs has size equal to the number of pairs.
        # Number of pairs in suffix: L*(L-1)/2 where L = len(suffix)
        # Number of cross pairs: j * L
        # Total = L*(L-1)/2 + j*L
        # If len(new_diffs) < this, there is a collision.
        expected_size = len(new_suffix)*(len(new_suffix)-1)//2 + j * len(new_suffix)
        if len(new_diffs) < expected_size:
            return False
            
        # Apply
        row[j:] = new_suffix
        used = (used - old_diffs) | new_diffs
        row_diffs[r] = _row_diffs(row)
        return True

    while time.time() < deadline and temperature > min_temp:
        r = rng.randrange(n_rows)
        j = rng.randrange(1, k + 1)
        
        old_scope = _scope(rows)
        
        # Save state
        saved_row = list(rows[r])
        saved_used = used.copy()
        saved_row_diffs_r = row_diffs[r].copy()
        
        if try_shift(r, j):
            new_scope = _scope(rows)
            delta = new_scope - old_scope
            if delta <= 0:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(x) for x in rows]
            else:
                if rng.random() < math.exp(-delta / temperature):
                    pass # Accept
                else:
                    # Revert
                    rows[r] = saved_row
                    used = saved_used
                    row_diffs[r] = saved_row_diffs_r
        temperature *= cooling

    return best_rows

def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)
    
    # Initial solution
    rows = _greedy(n_rows, k, rng)
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Local Search
    if time.time() < deadline - 10:
        improved = _local_search(rows, n_rows, k, deadline, rng)
        if _scope(improved) < best_scope:
            best = improved
            best_scope = _scope(best)
            
    # Optional: Restart if time remains?
    # For now, single run is safer.
    
    # Final validation
    if not _validate(best):
        best = _greedy(n_rows, k, rng)
        
    return best
