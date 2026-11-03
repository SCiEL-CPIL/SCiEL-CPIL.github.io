import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def _validate(rows, k):
    """Check if the set of rows is a valid DTS."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    """Return the maximum entry in any row."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _get_diffs(row):
    """Return the set of all pairwise differences in a row."""
    diffs = set()
    L = len(row)
    for i in range(L):
        for j in range(i + 1, L):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_initial(n_rows, k, rng):
    """
    Generate an initial valid DTS using a randomized greedy approach.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Randomly skip some values to avoid getting stuck in a bad local pattern
            if rng.random() < 0.1:
                x += rng.randint(1, 3)
            
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            
            if ok:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows

def _sa_optimize(n_rows, k, initial_rows, deadline, seed=42):
    """
    Simulated Annealing to minimize the maximum entry.
    Uses row reconstruction as the primary move.
    """
    rng = random.Random(seed)
    rows = [list(r) for r in initial_rows]
    L = k + 1
    
    # Initialize global difference set
    global_diffs = set()
    for r in rows:
        global_diffs.update(_get_diffs(r))
    
    current_max = _scope(rows)
    best_max = current_max
    best_rows = [r[:] for r in rows]
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random row to reconstruct
        r_idx = rng.randrange(n_rows)
        old_row = rows[r_idx]
        old_diffs = _get_diffs(old_row)
        
        # Remove old row's diffs from global set
        for d in old_diffs:
            global_diffs.discard(d)
        
        # Reconstruct the row
        # We want to find a valid row that minimizes the max entry.
        # We use a randomized greedy that tries to keep values low.
        new_row = [0]
        x = 0
        valid = True
        
        # We will try to construct the row up to L elements.
        # To allow for SA, we don't strictly minimize, but we bias towards smaller values.
        # We will pick the smallest valid x, but with some probability, we pick a larger one.
        
        for _ in range(L - 1):
            found = False
            # We search for the next value x
            # Start from last value + 1
            # To allow escaping local minima, we might start from a random offset? 
            # No, we must maintain increasing order.
            # We iterate x from last+1 upwards.
            # To make it stochastic, we can decide how far to look.
            
            # Let's try the smallest valid x first.
            # If we find it, we accept it with high probability.
            # To introduce randomness, we can skip the first few valid candidates.
            
            candidates = []
            limit = current_max + 20 # Don't search too far
            
            while x < limit:
                x += 1
                # Check if x can be appended to new_row
                ok = True
                for a in new_row:
                    d = x - a
                    if d in global_diffs:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
                    if len(candidates) > 5:
                        break
            
            if not candidates:
                valid = False
                break
            
            # Pick a candidate
            # Bias towards the smallest
            if rng.random() < 0.8:
                next_x = candidates[0]
            else:
                # Pick a random one from the top few
                next_x = candidates[rng.randint(0, min(4, len(candidates)-1))]
            
            new_row.append(next_x)
            x = next_x - 1 # Next iteration will increment to next_x
            
        if not valid:
            # Restore old row
            rows[r_idx] = old_row
            global_diffs.update(old_diffs)
            T *= cooling
            continue
            
        new_max = new_row[-1]
        
        # Calculate delta
        delta = new_max - current_max
        
        # Acceptance criterion
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta == 0:
                accept = True
            else:
                if rng.random() < math.exp(-delta / T):
                    accept = True
        
        if accept:
            # Update state
            rows[r_idx] = new_row
            new_diffs = _get_diffs(new_row)
            global_diffs.update(new_diffs)
            current_max = new_max
            
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
        else:
            # Restore old row
            rows[r_idx] = old_row
            global_diffs.update(old_diffs)
            
        T *= cooling
        
    return best_rows

def _polish(n_rows, k, rows, deadline, seed=99):
    """
    Final polish: Try to reduce the max entry by small amounts.
    Runs SA for a short time with a lower T.
    """
    rng = random.Random(seed)
    rows = [list(r) for r in rows]
    current_max = _scope(rows)
    
    # We will run a tight SA
    global_diffs = set()
    for r in rows:
        global_diffs.update(_get_diffs(r))
        
    T = 2.0
    T_min = 0.5
    cooling = 0.999
    
    start_time = time.time()
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r_idx = rng.randrange(n_rows)
        old_row = rows[r_idx]
        old_diffs = _get_diffs(old_row)
        
        for d in old_diffs:
            global_diffs.discard(d)
            
        new_row = [0]
        x = 0
        valid = True
        
        for _ in range(k):
            found = False
            limit = current_max + 5
            while x < limit:
                x += 1
                ok = True
                for a in new_row:
                    d = x - a
                    if d in global_diffs:
                        ok = False
                        break
                if ok:
                    # In polish, we strictly prefer the smallest
                    new_row.append(x)
                    found = True
                    break
            if not found:
                valid = False
                break
                
        if not valid:
            rows[r_idx] = old_row
            global_diffs.update(old_diffs)
            T *= cooling
            continue
            
        new_max = new_row[-1]
        delta = new_max - current_max
        
        if delta < 0:
            rows[r_idx] = new_row
            new_diffs = _get_diffs(new_row)
            global_diffs.update(new_diffs)
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
        elif delta == 0:
            # Sidestep to explore
            if rng.random() < 0.5:
                rows[r_idx] = new_row
                new_diffs = _get_diffs(new_row)
                global_diffs.update(new_diffs)
        else:
            # Reject worse
            rows[r_idx] = old_row
            global_diffs.update(old_diffs)
            
        T *= cooling
        
    return best_rows

def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0 # Safety margin
    
    # Phase 1: Initial Greedy
    rng = random.Random(42)
    best_rows = _greedy_initial(n_rows, k, rng)
    best_max = _scope(best_rows)
    
    # Phase 2: Main SA
    sa_deadline = min(deadline - 20.0, start + 100.0)
    sa_rows = _sa_optimize(n_rows, k, best_rows, sa_deadline, seed=42)
    sa_max = _scope(sa_rows)
    if sa_max < best_max:
        best_rows = sa_rows
        best_max = sa_max
        
    # Phase 3: Polish
    if time.time() < deadline - 5.0:
        polish_deadline = min(deadline - 2.0, time.time() + 15.0)
        polish_rows = _polish(n_rows, k, best_rows, polish_deadline, seed=99)
        polish_max = _scope(polish_rows)
        if polish_max < best_max:
            best_rows = polish_rows
            best_max = polish_max
            
    # Final Validation
    if not _validate(best_rows, k):
        # Fallback
        best_rows = _greedy_initial(n_rows, k, random.Random(0))
        
    return best_rows
