"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a combination of:
    1. Greedy construction with lookahead
    2. Local search / hill climbing to improve the solution
    3. Multiple restarts with different random seeds
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    
    # Total number of differences needed
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: largest entry must be at least such that
    # the number of possible differences 1..M is >= total_diffs
    # M*(M+1)/2 >= total_diffs
    import math
    min_scope = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    best_solution = None
    best_scope = float('inf')
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Greedy with various orderings
    # Strategy 2: Greedy with randomization
    # Strategy 3: Local search from greedy start
    
    # First, do a basic greedy to get an initial solution
    def greedy_construct(n_rows, k, randomize=False, rng=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Get initial greedy solution
    initial_rows = greedy_construct(n_rows, k)
    initial_scope = max(max(row) for row in initial_rows)
    best_solution = initial_rows
    best_scope = initial_scope
    
    # Try to improve with local search
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
        # Check all differences are distinct
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        return len(all_diffs) == len(set(all_diffs))
    
    def local_search(rows, n_rows, k, max_iter=5000, rng=None):
        """
        Local search: try to reduce the scope by adjusting elements.
        """
        if rng is None:
            rng = random.Random()
        
        current_rows = [row[:] for row in rows]
        current_scope = max(max(row) for row in current_rows)
        best_local_rows = [row[:] for row in current_rows]
        best_local_scope = current_scope
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random row and a random element (not the first) to modify
            row_idx = rng.randint(0, n_rows - 1)
            elem_idx = rng.randint(1, k)  # indices 1..k (not 0)
            
            old_val = current_rows[row_idx][elem_idx]
            
            # Try to move this element to a smaller value
            # We need to find a new value v such that:
            # 1. current_rows[row_idx][elem_idx - 1] < v < current_rows[row_idx][elem_idx + 1] (if exists)
            # 2. All new differences are not already used by other rows or other elements in this row
            
            # Compute current differences involving this element
            row = current_rows[row_idx]
            old_diffs = set()
            for i in range(len(row)):
                if i != elem_idx:
                    if row[elem_idx] > row[i]:
                        old_diffs.add(row[elem_idx] - row[i])
                    else:
                        old_diffs.add(row[i] - row[elem_idx])
            
            # Compute all differences not involving this element
            other_diffs = set()
            for r_idx in range(n_rows):
                r = current_rows[r_idx]
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        if r_idx == row_idx and (i == elem_idx or j == elem_idx):
                            continue
                        other_diffs.add(r[j] - r[i])
            
            # Try candidate values
            lower_bound = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            upper_bound = row[elem_idx + 1] - 1 if elem_idx < k else old_val
            
            # Try to reduce the value
            candidates = []
            # Try values from lower_bound to min(old_val - 1, upper_bound)
            search_start = lower_bound
            search_end = min(old_val - 1, upper_bound)
            
            if search_end >= search_start:
                # Try a subset of candidates for speed
                if search_end - search_start > 50:
                    # Sample candidates
                    step = max(1, (search_end - search_start) // 50)
                    candidates = list(range(search_start, search_end + 1, step))
                    candidates.append(search_end)
                else:
                    candidates = list(range(search_end, search_start - 1, -1))
            
            found = False
            for new_val in candidates:
                if new_val <= lower_bound - 1 or new_val >= upper_bound + 1:
                    continue
                
                # Check if new_val is valid
                new_diffs = set()
                for i in range(len(row)):
                    if i != elem_idx:
                        if new_val > row[i]:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                
                # Check no conflict with other_diffs
                if new_diffs & other_diffs:
                    continue
                
                # This is a valid move
                current_rows[row_idx][elem_idx] = new_val
                new_scope = max(max(r) for r in current_rows)
                
                if new_scope < best_local_scope:
                    best_local_rows = [r[:] for r in current_rows]
                    best_local_scope = new_scope
                
                found = True
                break
            
            if not found:
                # Try a random perturbation to escape local optima
                # Pick a random element and change it slightly
                if rng.random() < 0.1:
                    pert_row = rng.randint(0, n_rows - 1)
                    pert_elem = rng.randint(1, k)
                    old = current_rows[pert_row][pert_elem]
                    # Try a random nearby value
                    delta = rng.choice([-2, -1, 1, 2, 3, 5])
                    new = old + delta
                    r = current_rows[pert_row]
                    lb = r[pert_elem - 1] + 1 if pert_elem > 0 else 1
                    ub = r[pert_elem + 1] - 1 if pert_elem < k else best_local_scope
                    
                    if lb <= new <= ub:
                        # Check validity
                        new_diffs = set()
                        for i in range(len(r)):
                            if i != pert_elem:
                                if new > r[i]:
                                    new_diffs.add(new - r[i])
                                else:
                                    new_diffs.add(r[i] - new)
                        
                        other_diffs = set()
                        for r_idx in range(n_rows):
                            rr = current_rows[r_idx]
                            for i in range(len(rr)):
                                for j in range(i + 1, len(rr)):
                                    if r_idx == pert_row and (i == pert_elem or j == pert_elem):
                                        continue
                                    other_diffs.add(rr[j] - rr[i])
                        
                        if not (new_diffs & other_diffs):
                            current_rows[pert_row][pert_elem] = new
        
        return best_local_rows, best_local_scope
    
    # Run local search from the initial solution
    rng = random.Random(42)
    improved_rows, improved_scope = local_search(initial_rows, n_rows, k, max_iter=2000, rng=rng)
    
    if improved_scope < best_scope:
        best_scope = improved_scope
        best_solution = improved_rows
    
    # Try multiple restarts with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Create a randomized initial solution
        # Start with greedy, then randomly perturb
        rows = greedy_construct(n_rows, k)
        
        # Randomly adjust some elements
        for _ in range(50):
            if time.time() > deadline:
                break
            row_idx = rng.randint(0, n_rows - 1)
            elem_idx = rng.randint(1, k)
            row = rows[row_idx]
            old = row[elem_idx]
            delta = rng.randint(-5, 5)
            new = old + delta
            lb = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            ub = row[elem_idx + 1] - 1 if elem_idx < k else old + 10
            
            if lb <= new <= ub:
                # Check validity
                new_diffs = set()
                for i in range(len(row)):
                    if i != elem_idx:
                        if new > row[i]:
                            new_diffs.add(new - row[i])
                        else:
                            new_diffs.add(row[i] - new)
                
                other_diffs = set()
                for r_idx in range(n_rows):
                    r = rows[r_idx]
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            if r_idx == row_idx and (i == elem_idx or j == elem_idx):
                                continue
                            other_diffs.add(r[j] - r[i])
                
                if not (new_diffs & other_diffs):
                    row[elem_idx] = new
        
        # Verify validity
        if is_valid(rows, n_rows, k):
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_solution = [row[:] for row in rows]
        
        # Local search from this start
        improved_rows, improved_scope = local_search(rows, n_rows, k, max_iter=1000, rng=rng)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_solution = improved_rows
    
    # Final verification
    if best_solution is None:
        best_solution = initial_rows
    
    # Ensure the solution is valid
    if not is_valid(best_solution, n_rows, k):
        # Fall back to greedy
        best_solution = greedy_construct(n_rows, k)
    
    return best_solution
# EVOLVE-BLOCK-END
