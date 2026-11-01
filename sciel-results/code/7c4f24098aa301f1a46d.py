import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140  # seconds, leave margin for 150s limit
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def greedy_row(used_global, k):
        row = [0]
        local_used = set()
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_global) and not (new_diffs & local_used):
                row.append(x)
                local_used |= new_diffs
            x += 1
        return row

    def find_next_valid(row, local_used, used_global, start_x):
        x = start_x
        while True:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_global) and not (new_diffs & local_used):
                return x
            x += 1

    def randomized_greedy_row(used_global, k, skip_prob=0.2):
        row = [0]
        local_used = set()
        x = 1
        while len(row) < k + 1:
            x_valid = find_next_valid(row, local_used, used_global, x)
            if random.random() < skip_prob:
                # Try to skip to the next valid
                x_next = find_next_valid(row, local_used, used_global, x_valid + 1)
                x = x_next
            else:
                x = x_valid
            row.append(x)
            local_used |= {x - a for a in row[:-1]}
            x = x + 1
        return row

    # 1. Baseline Greedy
    used = set()
    baseline_rows = []
    for _ in range(n_rows):
        r = greedy_row(used, k)
        baseline_rows.append(r)
        used |= get_diffs(r)
    
    best_scope = max(max(r) for r in baseline_rows)
    best_rows = baseline_rows
    
    # 2. Beam Search with Branch and Bound
    beam_width = 100
    n_cands = 10
    
    # State: (rows, used_diffs)
    beam = [([], set())]
    
    for i in range(n_rows):
        new_beam = []
        
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        for rows, used_diffs in beam:
            current_max = max(max(r) for r in rows) if rows else 0
            
            # Prune if current max is already >= best_scope
            # Note: if rows is empty, current_max is 0.
            if current_max >= best_scope:
                continue
                
            # Generate candidates
            cands = []
            # 1. Greedy
            r_greedy = greedy_row(used_diffs, k)
            cands.append(r_greedy)
            
            # 2. Randomized
            for _ in range(n_cands - 1):
                if time.time() - start_time > time_limit:
                    break
                r_rand = randomized_greedy_row(used_diffs, k, skip_prob=0.3)
                if r_rand not in cands:
                    cands.append(r_rand)
                    
            for cand in cands:
                new_rows = rows + [cand]
                new_max = max(current_max, max(cand))
                
                # Prune if new max >= best_scope
                if new_max >= best_scope:
                    continue
                    
                new_used = used_diffs | get_diffs(cand)
                new_beam.append((new_rows, new_used))
                
        if i == n_rows - 1:
            # Last row, evaluate all
            for rows, _ in new_beam:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
            # We can break here as we have complete solutions
            break
        else:
            # Sort by max entry
            new_beam.sort(key=lambda state: max(max(r) for r in state[0]))
            beam = new_beam[:beam_width]
            
            # If beam is empty, we might have pruned everything.
            # This can happen if best_scope was updated to a very low value?
            # Or if no valid rows could be found under the bound.
            if not beam:
                break

    return best_rows
# EVOLVE-BLOCK-END
