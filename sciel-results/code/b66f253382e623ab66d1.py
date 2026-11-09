"""Difference triangle set (7,5): minimize largest entry with distinct intra-row differences."""
import time
import random
import copy

def _validate(rows, k):
    """Check that all positive intra-row differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    """Return the largest entry (scope) of the rows."""
    return max(max(r) for r in rows)

def _greedy_row(used, k, start=0):
    """Build one row greedily starting at 0, using smallest available marks."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_construction(n_rows, k):
    """Build all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _diffs_of_row(row):
    """Return the set of all positive intra-row differences."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _all_diffs(rows):
    """Return the union of all differences across all rows."""
    seen = set()
    for row in rows:
        seen |= _diffs_of_row(row)
    return seen

def _simulated_annealing(rows, k, n_rows, max_iters, start_temp, end_temp, rng):
    """
    Simulated annealing to minimize the scope.
    Moves: pick a random row and a random position (1..k), and either increment or decrement
    that mark by 1 (keeping the row strictly increasing). Accept if valid or with probability
    based on temperature.
    """
    n = k + 1
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    # Precompute differences for current state
    current_used = _all_diffs(current_rows)
    
    temp = start_temp
    total_iters = max_iters
    
    for it in range(total_iters):
        if it % 100 == 0:
            temp = start_temp * (end_temp / start_temp) ** (it / max(1, total_iters - 1))
        
        # Pick a random row and position
        r_idx = rng.randint(0, n_rows - 1)
        # Position must be 1..k (not 0, which is fixed)
        pos = rng.randint(1, k)
        
        old_val = current_rows[r_idx][pos]
        
        # Try increment or decrement
        delta = rng.choice([-1, 1])
        new_val = old_val + delta
        
        # Check bounds: must be > previous element and < next element
        if pos > 0 and new_val <= current_rows[r_idx][pos - 1]:
            continue
        if pos < k and new_val >= current_rows[r_idx][pos + 1]:
            continue
        if new_val < 0:
            continue
        
        # Compute new differences for this row
        old_row = current_rows[r_idx]
        new_row = old_row[:]
        new_row[pos] = new_val
        
        old_diffs = _diffs_of_row(old_row)
        new_diffs = _diffs_of_row(new_row)
        
        # Differences that are removed and added
        removed = old_diffs - new_diffs
        added = new_diffs - old_diffs
        
        # Check if any added difference is already in used (from other rows or same row's kept diffs)
        # used contains all diffs from all rows; we need to check added against (used - old_diffs)
        other_diffs = current_used - old_diffs
        if added & other_diffs:
            continue
        
        # Valid move
        # Apply
        current_rows[r_idx] = new_row
        current_used = (current_used - old_diffs) | new_diffs
        
        new_scope = _scope(current_rows)
        
        if new_scope <= current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Accept with probability exp(-(new_scope - current_scope)/temp)
            delta_e = new_scope - current_scope
            if temp > 0 and rng.random() < (1.0 / (1.0 + delta_e * delta_e / (temp * temp + 1e-10))):
                # Accept
                pass
            else:
                # Reject: revert
                current_rows[r_idx] = old_row
                current_used = (current_used - new_diffs) | old_diffs
                continue
    
    return best_rows, best_scope

def construct(n_rows, k):
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # Start with greedy construction
    rows = _greedy_construction(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    rng = random.Random(42)
    
    # Simulated annealing with multiple restarts
    start_temp = 2.0
    end_temp = 0.01
    
    while time.time() < deadline:
        # Run SA from current best (or slightly perturbed)
        sa_rows, sa_scope = _simulated_annealing(
            best_rows, k, n_rows,
            max_iters=5000,
            start_temp=start_temp,
            end_temp=end_temp,
            rng=rng
        )
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [list(r) for r in sa_rows]
        
        # Occasionally restart from greedy with different randomization
        # by perturbing the greedy result
        if time.time() < deadline:
            perturbed = [list(r) for r in best_rows]
            # Small random perturbations
            for _ in range(5):
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)
                old_val = perturbed[r_idx][pos]
                new_val = old_val + rng.choice([-1, 0, 1])
                if pos > 0 and new_val <= perturbed[r_idx][pos - 1]:
                    continue
                if pos < k and new_val >= perturbed[r_idx][pos + 1]:
                    continue
                if new_val < 0:
                    continue
                perturbed[r_idx][pos] = new_val
            
            # Check validity
            if _validate(perturbed, k):
                pert_scope = _scope(perturbed)
                if pert_scope < best_scope:
                    best_scope = pert_scope
                    best_rows = [list(r) for r in perturbed]
    
    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(7, 5)
    print("Scope:", _scope(rows))
    print("Valid:", _validate(rows, 5))
    for r in rows:
        print(r)
