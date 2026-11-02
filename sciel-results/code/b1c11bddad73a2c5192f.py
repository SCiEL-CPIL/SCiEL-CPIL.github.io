"""Difference triangle set (n_rows, k) via randomized greedy + simulated annealing."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill (and 150s wall)
    
    # Total number of differences needed
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on scope: need at least total_diffs distinct positive differences
    # Max difference in a row with max value m is m, so we need m*(m+1)/2 >= total_diffs
    # => m >= approx sqrt(2 * total_diffs)
    # But this is a weak bound; the counting bound mentioned is 112 for (4,7)
    
    def greedy_row(used, start_x=1):
        """Build one row greedily using the used set."""
        row = [0]
        x = start_x
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
            x += 1
        return row
    
    def all_rows_greedy(used, start_x=1):
        """Build all rows greedily."""
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, start_x)
            rows.append(row)
        return rows
    
    def scope_of(rows):
        """Return the largest entry across all rows."""
        return max(max(r) for r in rows)
    
    def validate(rows):
        """Check that all differences are distinct."""
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Initial solution: deterministic greedy
    best_rows = all_rows_greedy(set(), 1)
    best_scope = scope_of(best_rows)
    
    # Simulated annealing / multi-start search
    # Strategy: 
    # 1. Try many randomized greedy constructions
    # 2. Local search: pick a row, remove it, rebuild it with different randomization
    # 3. Accept worse solutions with probability exp(-delta/T)
    
    def randomized_greedy_row(used, start_x=1, step_prob=0.3):
        """Build one row with some randomization in step sizes."""
        row = [0]
        x = start_x
        while len(row) < k + 1:
            # Try current x
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
                # Randomly jump ahead sometimes
                if random.random() < step_prob:
                    x += random.randint(1, 3)
                else:
                    x += 1
            else:
                x += 1
        return row
    
    def randomized_greedy_all(used, start_x=1, step_prob=0.3):
        """Build all rows with randomization."""
        rows = []
        for _ in range(n_rows):
            row = randomized_greedy_row(used, start_x, step_prob)
            rows.append(row)
        return rows
    
    def rebuild_one_row(rows, row_idx, used, start_x=1, step_prob=0.3):
        """Remove one row and rebuild it."""
        old_row = rows[row_idx]
        # Remove old row's differences from used
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                used.discard(old_row[j] - old_row[i])
        
        # Rebuild
        new_row = randomized_greedy_row(used, start_x, step_prob)
        
        # Add new row's differences
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                used.add(new_row[j] - new_row[i])
        
        rows[row_idx] = new_row
        return rows
    
    # Multi-start with simulated annealing
    T = 10.0
    cooling = 0.995
    best_scope = scope_of(best_rows)
    current_scope = best_scope
    current_rows = [r[:] for r in best_rows]
    
    # Build used set for current solution
    def build_used(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    current_used = build_used(current_rows)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random row to rebuild
        row_idx = random.randint(0, n_rows - 1)
        
        # Save state
        old_row = current_rows[row_idx][:]
        old_scope = current_scope
        
        # Rebuild with randomization
        new_rows = rebuild_one_row(current_rows, row_idx, current_used, 
                                    start_x=1, step_prob=0.2 + random.random() * 0.3)
        new_scope = scope_of(new_rows)
        
        # Accept if better, or with probability exp(-(new_scope - old_scope)/T)
        if new_scope < old_scope:
            current_rows = new_rows
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            delta = new_scope - old_scope
            if delta > 0 and random.random() < pow(2.718281828, -delta / max(T, 0.1)):
                current_rows = new_rows
                current_scope = new_scope
            else:
                # Revert
                current_rows[row_idx] = old_row
                # Rebuild used set properly
                current_used = build_used(current_rows)
        
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1
        
        # Periodically restart with fresh randomized greedy
        if iteration % 50 == 0:
            fresh_rows = randomized_greedy_all(set(), 1, 0.2 + random.random() * 0.3)
            fresh_scope = scope_of(fresh_rows)
            if fresh_scope < best_scope:
                best_scope = fresh_scope
                best_rows = [r[:] for r in fresh_rows]
                current_rows = [r[:] for r in fresh_rows]
                current_scope = fresh_scope
                current_used = build_used(current_rows)
                T = 5.0  # Reheat
    
    # Final polish: try to improve each row individually with tighter search
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() >= deadline - 10:
                break
            row_idx = random.randint(0, n_rows - 1)
            old_row = best_rows[row_idx][:]
            
            # Try to rebuild this row more tightly
            used = build_used(best_rows)
            # Remove old row's diffs
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    used.discard(old_row[j] - old_row[i])
            
            # Try multiple random rebuilds
            best_new_row = None
            best_new_scope = scope_of(best_rows)
            for attempt in range(20):
                trial_row = randomized_greedy_row(used, 1, 0.1 + random.random() * 0.2)
                trial_rows = best_rows[:]
                trial_rows[row_idx] = trial_row
                trial_scope = scope_of(trial_rows)
                if trial_scope < best_new_scope:
                    best_new_scope = trial_scope
                    best_new_row = trial_row
            
            if best_new_row is not None:
                best_rows[row_idx] = best_new_row
                best_scope = best_new_scope
    
    return best_rows
# EVOLVE-BLOCK-END
