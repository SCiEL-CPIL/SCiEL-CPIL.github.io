import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set using randomized greedy initialization 
    and simulated annealing local search.
    """
    deadline = time.time() + 140.0  # 10s safety margin
    best_rows = None
    best_scope = float('inf')
    
    # Helper to compute scope
    def get_scope(rows):
        return max(max(row) for row in rows) if rows else 0

    # Helper to get all differences of a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Randomized Greedy Construction
    def randomized_greedy(n_rows, k, randomness=0.3):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Find valid candidates
                candidates = []
                x = row[-1] + 1
                # Limit search range to avoid infinite loops if stuck, 
                # though for small k it should find quickly
                max_search = row[-1] + 200 
                while x < max_search:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        if len(candidates) > 20:
                            break
                    x += 1
                
                if not candidates:
                    return None # Failed to construct
                    
                # Choose candidate
                if random.random() < randomness:
                    # Pick from top 5 candidates to introduce variation
                    top_candidates = candidates[:min(5, len(candidates))]
                    x = random.choice(top_candidates)
                else:
                    x = candidates[0]
                    
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        return rows

    # Simulated Annealing Refinement
    def refine(rows, time_limit):
        if not rows:
            return rows, 0
            
        # Deep copy
        cur_rows = [list(r) for r in rows]
        used = set()
        for r in cur_rows:
            used.update(get_row_diffs(r))
            
        cur_scope = get_scope(cur_rows)
        best_local_rows = [list(r) for r in cur_rows]
        best_local_scope = cur_scope
        
        T = 10.0
        min_T = 0.1
        cooling = 0.95
        
        start_time = time.time()
        
        while T > min_T and time.time() < time_limit:
            move_type = random.random()
            
            new_rows = [list(r) for r in cur_rows]
            new_used = set(used)
            valid_move = True
            
            if move_type < 0.5:
                # Shift element in a random row
                r_idx = random.randint(0, n_rows - 1)
                # Choose index > 0
                i_idx = random.randint(1, k) 
                
                old_val = new_rows[r_idx][i_idx]
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                
                # Check strict increase
                if i_idx > 0 and new_val <= new_rows[r_idx][i_idx - 1]:
                    valid_move = False
                if i_idx < k and new_val >= new_rows[r_idx][i_idx + 1]:
                    valid_move = False
                
                if valid_move:
                    # Remove old diffs for this row
                    old_diffs = get_row_diffs(new_rows[r_idx])
                    new_used -= old_diffs
                    
                    # Update row
                    new_rows[r_idx][i_idx] = new_val
                    
                    # Check new diffs
                    new_diffs = get_row_diffs(new_rows[r_idx])
                    if new_diffs & new_used:
                        valid_move = False
                    else:
                        new_used |= new_diffs
            else:
                # Regenerate a random row
                r_idx = random.randint(0, n_rows - 1)
                
                # Remove diffs of row r_idx
                old_diffs = get_row_diffs(new_rows[r_idx])
                new_used -= old_diffs
                
                # Reconstruct row r_idx greedily
                new_row = [0]
                success = True
                while len(new_row) < k + 1:
                    x = new_row[-1] + 1
                    found = False
                    # Search limit
                    for _ in range(100):
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & new_used):
                            new_row.append(x)
                            new_used.update(new_diffs)
                            found = True
                            break
                        x += 1
                    if not found:
                        success = False
                        break
                
                if success:
                    new_rows[r_idx] = new_row
                else:
                    valid_move = False
                    # Revert used set
                    new_used = set(used)
                    for r in cur_rows:
                        new_used.update(get_row_diffs(r))
            
            if valid_move:
                new_scope = get_scope(new_rows)
                
                # Acceptance criterion
                if new_scope < cur_scope:
                    # Accept
                    cur_rows = new_rows
                    used = new_used
                    cur_scope = new_scope
                    if new_scope < best_local_scope:
                        best_local_scope = new_scope
                        best_local_rows = [list(r) for r in new_rows]
                else:
                    # Probabilistic acceptance
                    prob = math.exp(-(new_scope - cur_scope) / T)
                    if random.random() < prob:
                        cur_rows = new_rows
                        used = new_used
                        cur_scope = new_scope
            
            T *= cooling
            
        return best_local_rows, best_local_scope

    # Main Loop
    while time.time() < deadline:
        # Generate initial solution
        rows = randomized_greedy(n_rows, k, randomness=0.5)
        if rows is None:
            continue
            
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
        # Refine if time permits
        if time.time() < deadline - 2:
            # Allocate remaining time to refinement, capped at 5s per iteration
            time_left = deadline - time.time()
            refine_time = min(time_left - 1, 5.0) 
            if refine_time > 0.1:
                refined_rows, refined_scope = refine(rows, time.time() + refine_time)
                if refined_scope < best_scope:
                    best_scope = refined_scope
                    best_rows = refined_rows

    return best_rows

# Fallback if no valid solution found (should not happen)
if __name__ == "__main__":
    # Simple test
    pass
