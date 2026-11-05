import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for 150s limit

    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row, used_diffs):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = get_row_diffs(row)
        # Check internal uniqueness
        if len(diffs) != len(row) * (len(row) - 1) // 2:
            return False
        # Check global uniqueness
        if diffs & used_diffs:
            return False
        return True

    def generate_row_greedy(used_diffs, max_scope_hint):
        """
        Greedily construct a single row given the used differences.
        Tries to keep values as small as possible.
        """
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Start from the smallest possible value
            min_val = row[-1] + 1
            # We don't strictly need max_scope_hint for generation, but it helps bound search if we want to be strict.
            # However, for SA, we want to find *any* valid row, preferably small.
            # Let's search up to a reasonable limit. If we fail, we return None.
            # A safe upper bound for the next element is current_max + (k+1) or something similar, but let's just scan.
            # To be efficient, we limit the scan. If we can't find one quickly, we fail.
            for x in range(min_val, max_scope_hint + 1 if max_scope_hint else 500):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs) and len(new_diffs) == len(row):
                    row.append(x)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        return row

    def initial_solution():
        """
        Create an initial valid solution using greedy construction.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            # Try to generate a row
            row = None
            # We might need to retry with a higher scope if greedy fails, 
            # but for (7,5) scope ~100-120, greedy usually works.
            for attempt in range(10):
                row = generate_row_greedy(used, 200)
                if row:
                    break
            if row is None:
                # Fallback: deterministic greedy without scope limit
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used) and len(new_diffs) == len(row):
                            break
                        x += 1
                    row.append(x)
                # Update used
                used.update(get_row_diffs(row))
            else:
                used.update(get_row_diffs(row))
            rows.append(row)
        return rows

    def get_scope(rows):
        return max(max(r) for r in rows)

    def local_search(rows, time_budget):
        """
        Perform local search to minimize the scope.
        """
        current_rows = [list(r) for r in rows]
        current_scope = get_scope(current_rows)
        improved = True
        start_ls = time.time()
        
        while improved and time.time() - start_ls < time_budget:
            improved = False
            # Shuffle row order to avoid bias
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            for i in row_order:
                if time.time() - start_ls >= time_budget:
                    break
                row = current_rows[i]
                # Try to improve each element from right to left
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    min_val = row[pos - 1] + 1
                    if current_val == min_val:
                        continue
                    
                    # Calculate diffs of other rows
                    other_diffs = set()
                    for j, r in enumerate(current_rows):
                        if j != i:
                            other_diffs.update(get_row_diffs(r))
                    
                    # Try to reduce current_val
                    found = False
                    for v in range(min_val, current_val):
                        # Check if v is valid for this position
                        # 1. Strictly increasing: v > row[pos-1] (guaranteed by range)
                        # 2. Unique diffs within row
                        # 3. No conflict with other rows
                        candidate_row = list(row)
                        candidate_row[pos] = v
                        
                        # Check internal validity
                        valid = True
                        cand_diffs = get_row_diffs(candidate_row)
                        if len(cand_diffs) != len(candidate_row) * (len(candidate_row) - 1) // 2:
                            continue
                        if cand_diffs & other_diffs:
                            continue
                        
                        # Accept
                        current_rows[i] = candidate_row
                        new_scope = get_scope(current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                        improved = True
                        found = True
                        break
                    if found:
                        break # Move to next row
        return current_rows, current_scope

    # --- Main Logic ---
    
    # 1. Initial Solution
    rows = initial_solution()
    best_rows = [list(r) for r in rows]
    best_scope = get_scope(best_rows)
    
    # 2. Local Search on Initial
    if time.time() - start_time < time_limit - 10:
        ls_rows, ls_scope = local_search(best_rows, time_limit - (time.time() - start_time) - 10)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows
    
    # 3. Simulated Annealing
    # We will perturb rows and try to improve.
    # State: rows
    # Cost: Scope
    
    # Parameters
    T_start = 10.0
    T_end = 0.1
    alpha = 0.995 # Cooling rate
    
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    T = T_start
    
    while time.time() - start_time < time_limit - 5:
        # Check time
        if time.time() - start_time > time_limit - 2:
            break
            
        # Generate a move
        move_type = random.random()
        if move_type < 0.7:
            # Perturbation: Regenerate a random row
            i = random.randint(0, n_rows - 1)
            # Remove row i from used diffs
            row_diffs = get_row_diffs(current_rows[i])
            # We need to rebuild the used set without row i
            used = set()
            for j, r in enumerate(current_rows):
                if j != i:
                    used.update(get_row_diffs(r))
            
            # Try to generate a new row
            new_row = generate_row_greedy(used, best_scope + 10) # Allow slightly larger
            if new_row is None:
                # If greedy fails, try a random walk or just skip
                # For simplicity, skip if greedy fails, as it implies tight constraints
                continue
                
            # Calculate new scope
            new_scope = max(get_scope([r for j, r in enumerate(current_rows) if j != i]), max(new_row))
            
            # Metropolis criterion
            delta = new_scope - current_scope
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_rows[i] = new_row
                current_scope = new_scope
                
                # Quick local search on the modified row? No, let SA handle it.
                # But we can try to reduce the new row immediately.
                # Let's do a fast local search on the whole set if we improved significantly
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
                    
        else:
            # Perturbation: Adjust a random element in a random row downwards
            i = random.randint(0, n_rows - 1)
            pos = random.randint(1, k) # 1 to k (index in row)
            current_val = current_rows[i][pos]
            min_val = current_rows[i][pos - 1] + 1
            
            if current_val == min_val:
                continue
                
            # Try a random smaller value
            v = random.randint(min_val, current_val - 1)
            
            # Check validity
            candidate_row = list(current_rows[i])
            candidate_row[pos] = v
            
            # Check internal
            cand_diffs = get_row_diffs(candidate_row)
            if len(cand_diffs) != len(candidate_row) * (len(candidate_row) - 1) // 2:
                continue
                
            # Check global
            other_diffs = set()
            for j, r in enumerate(current_rows):
                if j != i:
                    other_diffs.update(get_row_diffs(r))
                    
            if cand_diffs & other_diffs:
                continue
                
            # Valid move
            new_scope = get_scope([r if j != i else candidate_row for j, r in enumerate(current_rows)])
            
            delta = new_scope - current_scope
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_rows[i] = candidate_row
                current_scope = new_scope
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
        
        # Cool down
        T *= alpha
        if T < T_end:
            # If cooled down, do a final local search and break or restart?
            # Let's do a final local search and break if no time
            if time.time() - start_time < time_limit - 10:
                ls_rows, ls_scope = local_search(current_rows, time_limit - (time.time() - start_time) - 10)
                if ls_scope < best_scope:
                    best_scope = ls_scope
                    best_rows = ls_rows
            break

    # Final local search to polish
    if time.time() - start_time < time_limit - 2:
        ls_rows, ls_scope = local_search(best_rows, time_limit - (time.time() - start_time) - 2)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows

    return best_rows

if __name__ == "__main__":
    # Test with (7, 5)
    res = construct(7, 5)
    print("Scope:", max(max(r) for r in res))
    for r in res:
        print(r)
