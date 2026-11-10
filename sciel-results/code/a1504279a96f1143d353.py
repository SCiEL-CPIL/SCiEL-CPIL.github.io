import time
import random
import math

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140  # Safety margin

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None, randomize=False):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline, randomize=False):
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        
        candidates = list(range(last + 1, upper + 1))
        if randomize:
            random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize)
            if res is not None:
                return res
        return None

    def global_max(rows_list):
        m = 0
        for r in rows_list:
            if r[-1] > m:
                m = r[-1]
        return m

    # Phase 1: Greedy Baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        rows.append(row)
        used |= row_diffs(row)
    
    current_max = global_max(rows)
    best_rows = rows[:]
    best_max = current_max

    # Phase 2: Simulated Annealing with multi-row moves
    T = 15.0
    cooling = 0.997
    min_T = 0.05

    def generate_candidate(i, cap_override=None, randomize=True):
        """Generate a candidate row for index i, avoiding all other rows' diffs."""
        used_without_i = set(used)
        used_without_i.difference_update(row_diffs(rows[i]))
        
        cap = cap_override if cap_override is not None else current_max
        if cap < 0:
            return None
        
        # Allow slightly larger cap to escape local minima
        search_cap = cap + random.randint(0, 3)
        
        row_deadline = min(deadline - 5, time.time() + 0.4)
        cand = dfs_row([0], used_without_i, search_cap, row_deadline, randomize=randomize)
        return cand

    # Main SA Loop
    sa_iter = 0
    while time.time() < deadline - 15:
        if T < min_T:
            break
        
        sa_iter += 1
        
        # Choose move type: single row replace (80%) or two-row swap (20%)
        move_type = random.random()
        
        if move_type < 0.8:
            # Single row replacement
            i = random.randint(0, n_rows - 1)
            cand = generate_candidate(i, randomize=True)
            
            if cand is not None:
                # Compute new global max
                new_rows = rows[:]
                new_rows[i] = cand
                new_max = global_max(new_rows)
                delta = new_max - current_max
                
                if delta <= 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(-delta / T)
                    except:
                        prob = 0.0
                    accept = random.random() < prob
                
                if accept:
                    used.difference_update(row_diffs(rows[i]))
                    rows[i] = cand
                    used |= row_diffs(cand)
                    current_max = new_max
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = rows[:]
        else:
            # Two-row swap: replace rows[i] and rows[j] simultaneously
            if n_rows < 2:
                T *= cooling
                continue
            
            i, j = random.sample(range(n_rows), 2)
            
            # First, find candidate for i avoiding all rows except j
            used_without_j = set(used)
            used_without_j.difference_update(row_diffs(rows[j]))
            
            cap_i = current_max + random.randint(0, 3)
            row_deadline = min(deadline - 5, time.time() + 0.3)
            cand_i = dfs_row([0], used_without_j, cap_i, row_deadline, randomize=True)
            
            if cand_i is None:
                T *= cooling
                continue
            
            # Now find candidate for j avoiding all rows except i's new candidate
            # used without j and without i's old row, but with i's new row
            used_for_j = set(used)
            used_for_j.difference_update(row_diffs(rows[j]))
            used_for_j.difference_update(row_diffs(rows[i]))
            used_for_j.update(row_diffs(cand_i))
            
            cap_j = current_max + random.randint(0, 3)
            row_deadline = min(deadline - 5, time.time() + 0.3)
            cand_j = dfs_row([0], used_for_j, cap_j, row_deadline, randomize=True)
            
            if cand_j is None:
                T *= cooling
                continue
            
            # Check validity: cand_i and cand_j must not share any diffs
            diffs_i = row_diffs(cand_i)
            diffs_j = row_diffs(cand_j)
            if diffs_i & diffs_j:
                T *= cooling
                continue
            
            # Compute new global max
            new_rows = rows[:]
            new_rows[i] = cand_i
            new_rows[j] = cand_j
            new_max = global_max(new_rows)
            delta = new_max - current_max
            
            if delta <= 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                accept = random.random() < prob
            
            if accept:
                used.difference_update(row_diffs(rows[i]))
                used.difference_update(row_diffs(rows[j]))
                rows[i] = cand_i
                rows[j] = cand_j
                used |= diffs_i
                used |= diffs_j
                current_max = new_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows[:]
        
        T *= cooling

    # Phase 3: Final Polish - iterative shrinking with multiple passes
    final_rows = best_rows[:]
    final_used = set()
    for r in final_rows:
        final_used |= row_diffs(r)
    
    # Multiple passes of deterministic shrinking
    for polish_pass in range(3):
        if time.time() > deadline - 3:
            break
        
        improved = False
        for i in range(n_rows):
            if time.time() > deadline - 2:
                break
            cur_last = final_rows[i][-1]
            if cur_last <= 0:
                continue
            
            used_without = set(final_used)
            used_without.difference_update(row_diffs(final_rows[i]))
            
            row_deadline = min(deadline - 2, time.time() + 0.8)
            
            # Try to find a smaller row
            cand = dfs_row([0], used_without, cur_last - 1, row_deadline, randomize=False)
            if cand is not None:
                final_used.difference_update(row_diffs(final_rows[i]))
                final_rows[i] = cand
                final_used |= row_diffs(cand)
                improved = True
        
        if not improved:
            break
    
    # Final validation and return
    return final_rows
