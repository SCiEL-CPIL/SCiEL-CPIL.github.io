import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization with correct internal diff checking.
    """
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Calculate internal diffs of current row
            internal_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    internal_diffs.add(row[j] - row[i])
            
            # Search for next element
            for x in range(current_max + 1, current_max + 500): 
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    if d in internal_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                    current_max = x
                    found = True
                    break
            
            if not found:
                # Fallback
                x = current_max + 500
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        # Check internal
                        int_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                int_diffs.add(row[j]-row[i])
                        if not any(d in int_diffs for d in new_diffs):
                             row.append(x)
                             for d in new_diffs:
                                 used_diffs.add(d)
                             current_max = x
                             break
                    x += 1
        rows.append(row)
        
    return rows

def _compute_diffs_for_row(row):
    """Return a list of differences for a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialization
    current_rows = _greedy_init(n_rows, k)
    
    # Pre-compute diff counts for the whole system
    diff_counts = {}
    for r in current_rows:
        for d in _compute_diffs_for_row(r):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    def get_energy(rows, diff_counts):
        max_val = max(max(r) for r in rows)
        conflicts = sum(1 for count in diff_counts.values() if count > 1)
        return max_val + 1000 * conflicts

    current_energy = get_energy(current_rows, diff_counts)
    best_rows = [r[:] for r in current_rows]
    best_energy = current_energy
    current_global_max = max(max(r) for r in current_rows)
    
    T = 50.0
    T_min = 0.01
    cool = 0.9995
    
    row_len = k + 1
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r_idx = random.randrange(n_rows)
        row = current_rows[r_idx]
        
        # Pick element index 1..k
        j_idx = random.randrange(1, row_len)
        
        # Pick delta
        delta = random.choice([-2, -1, 1, 2])
        
        old_val = row[j_idx]
        new_val = old_val + delta
        
        # Check monotonicity
        if (j_idx > 0 and new_val <= row[j_idx - 1]) or (j_idx < row_len - 1 and new_val >= row[j_idx + 1]):
            T *= cool
            continue
            
        # Calculate diffs that change
        old_diffs = set()
        for i in range(row_len):
            if i != j_idx:
                d = abs(row[j_idx] - row[i])
                old_diffs.add(d)
                
        new_diffs = set()
        for i in range(row_len):
            if i != j_idx:
                d = abs(new_val - row[i])
                new_diffs.add(d)
        
        removed_diffs = old_diffs - new_diffs
        added_diffs = new_diffs - old_diffs
        
        delta_conflicts = 0
        
        for d in added_diffs:
            c = diff_counts.get(d, 0)
            if c >= 1:
                delta_conflicts += 1
        
        for d in removed_diffs:
            c = diff_counts.get(d, 0)
            if c == 2:
                delta_conflicts -= 1
            
        # Update diff_counts
        for d in added_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        for d in removed_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Update row
        row[j_idx] = new_val
        
        # Update global max
        new_row_max = max(row)
        if new_row_max > current_global_max:
            current_global_max = new_row_max
        elif new_row_max < current_global_max:
            # Check if any other row has this max
            # Simple recompute for correctness
            current_global_max = max(max(r) for r in current_rows)
            
        new_energy = current_global_max + 1000 * sum(1 for count in diff_counts.values() if count > 1)
        
        delta_energy = new_energy - current_energy
        
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject: revert changes
            row[j_idx] = old_val
            for d in added_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in removed_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            # Revert global max
            if new_val > old_val and new_val == current_global_max:
                 current_global_max = max(max(r) for r in current_rows)
            elif old_val > new_val and old_val == current_global_max:
                 current_global_max = max(max(r) for r in current_rows)
                 
        T *= cool
        
    return best_rows

# EVOLVE-BLOCK-END
