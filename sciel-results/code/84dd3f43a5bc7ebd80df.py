import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.15, window=30, seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    def local_search(rows, time_limit):
        """
        Iterated Local Search with targeted descent, random perturbation, and row reconstruction.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def try_decrease_element(best_rows, r_idx, c_idx):
            """Try to decrease a specific element. Returns True if improved."""
            curr_val = best_rows[r_idx][c_idx]
            prev_v = best_rows[r_idx][c_idx-1]
            
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            row = best_rows[r_idx]
            for i in range(c_idx):
                remove_set.add(curr_val - row[i])
            for j in range(c_idx + 1, len(row)):
                remove_set.add(row[j] - curr_val)
            
            fixed_diffs = all_diffs - remove_set
            
            for new_val in range(curr_val - 1, prev_v, -1):
                valid = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                for j in range(c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left = [new_val - row[i] for i in range(c_idx)]
                new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                if len(set(new_left + new_right)) != len(new_left + new_right):
                    continue
                
                row[c_idx] = new_val
                return True
            return False

        def reconstruct_row(best_rows, r_idx):
            """Reconstruct a single row to escape local minima. Returns True if successful."""
            row = best_rows[r_idx][:]
            # Remove diffs for this row
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            for i in range(len(row)):
                for j in range(i):
                    remove_set.add(row[j] - row[i])
            fixed_diffs = all_diffs - remove_set
            
            # Rebuild row
            new_row = [0]
            valid = True
            for _ in range(row_len - 1):
                last = new_row[-1]
                found = False
                x = last + 1
                while x <= best_scope + 10:
                    ok = True
                    for a in new_row:
                        if (x - a) in fixed_diffs:
                            ok = False
                            break
                    if ok:
                        # Check internal consistency
                        internal_ok = True
                        for i in range(len(new_row)):
                            d = x - new_row[i]
                            for j in range(i):
                                if d == new_row[i] - new_row[j]:
                                    internal_ok = False
                                    break
                            if not internal_ok:
                                break
                        if internal_ok:
                            new_row.append(x)
                            found = True
                            break
                    x += 1
                if not found:
                    valid = False
                    break
            
            if valid and len(new_row) == row_len:
                best_rows[r_idx] = new_row
                return True
            return False

        stuck_counter = 0
        no_improve_counter = 0

        while time.time() < start + time_limit:
            # 1. Targeted Descent: Try to reduce the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break
            
            found_improvement = False
            
            # Try to decrease the max element
            if try_decrease_element(best_rows, max_r_idx, max_c_idx):
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                found_improvement = True
            
            if found_improvement:
                continue

            # 2. Try to decrease other elements
            improved_any = False
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if time.time() > start + time_limit:
                        break
                    if try_decrease_element(best_rows, r_idx, c_idx):
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            stuck_counter = 0
                            no_improve_counter = 0
                        else:
                            no_improve_counter += 1
                        improved_any = True
                        break
                if improved_any:
                    break
            
            if improved_any:
                continue

            # 3. Perturbation: Randomly move an element
            if time.time() < start + time_limit:
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                row = best_rows[r_idx]
                curr_val = row[c_idx]
                prev_v = row[c_idx-1]
                next_v = row[c_idx+1] if c_idx < row_len-1 else curr_val + 50
                
                all_diffs = get_all_diffs(best_rows)
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - curr_val)
                fixed_diffs = all_diffs - remove_set
                
                # Try a random new value
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50
                
                if low <= high:
                    for _ in range(10):
                        cand = random.randint(low, high)
                        if cand == curr_val:
                            continue
                        
                        valid = True
                        for i in range(c_idx):
                            d = cand - row[i]
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - cand
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_left = [cand - row[i] for i in range(c_idx)]
                        new_right = [row[j] - cand for j in range(c_idx + 1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        
                        row[c_idx] = cand
                        improved_any = True
                        break
                
                if improved_any:
                    continue

            # 4. Row Reconstruction if stuck
            if no_improve_counter > 5 or stuck_counter > 10:
                r_idx = random.randint(0, n_rows - 1)
                if reconstruct_row(best_rows, r_idx):
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        stuck_counter = 0
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                    continue
            
            no_improve_counter += 1
            stuck_counter += 1

        return best_rows, best_scope

    def beam_search(n_rows, k, max_time, initial_scope=None):
        """
        Beam search where we build rows incrementally, interleaving rows to minimize max scope.
        State: (rows, used_diffs)
        """
        start = time.time()
        beam_width = 50
        
        # Initial state: all rows have [0]
        initial_rows = [[0] for _ in range(n_rows)]
        initial_used = set()
        
        # We need to track the "next element" for each row implicitly by the row content.
        # We expand one element at a time across all rows.
        
        # To make it efficient, we keep a list of states.
        # State: (rows, used_diffs, max_val)
        
        if initial_scope is None:
            # Estimate a loose upper bound
            initial_scope = 200 
        
        # We will try to find a solution with scope < initial_scope
        # We expand the beam.
        
        current_beam = [(initial_rows, initial_used, 0)]
        
        # We need to expand until all rows are full.
        # Total elements to add: n_rows * k (since 0 is already there)
        
        for step in range(n_rows * k):
            next_beam = []
            
            for rows, used_diffs, curr_max in current_beam:
                # Determine which rows need expansion
                # We want to pick a row to add the next element to.
                # Heuristic: Pick the row with the smallest current max? Or random?
                # Let's pick the row with the smallest last element to keep them balanced, 
                # or just iterate through rows and expand the first one that has room?
                # Actually, for beam search, we should expand ALL possible choices for the NEXT move.
                # A "move" is choosing a row i and a value x for row i.
                
                # This branching factor is huge. 
                # Better approach: Pick ONE row to expand at each step? 
                # No, that's just sequential. 
                # Let's try: At each step, for each row that is not full, try to add the smallest valid x.
                # But that's greedy.
                
                # Let's try: For each state in beam, generate children by adding the smallest valid x to EACH row that is not full.
                # This keeps the branching factor at n_rows.
                
                for r_idx in range(n_rows):
                    if len(rows[r_idx]) >= row_len:
                        continue
                    
                    current_row = rows[r_idx]
                    last_val = current_row[-1]
                    
                    # Find smallest valid x
                    x = last_val + 1
                    found = False
                    while x <= initial_scope:
                        valid = True
                        for a in current_row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            # Check internal consistency (diffs within the row are unique)
                            # Since we build incrementally, we just need to check against existing elements in current_row
                            # which is done above.
                            # But wait, d = x - a. If a is in current_row, is it possible that x-a duplicates a diff already in current_row?
                            # Yes. e.g. row [0, 2], x=4. d=4, 2. 2 is already in row (4-2=2, 2-0=2). 
                            # So we must check against all existing elements in the row.
                            # The loop `for a in current_row` does this.
                            # However, we must also ensure that the new diffs don't conflict with each other?
                            # No, the diffs are x-a1, x-a2... These are all distinct because a1 != a2.
                            # So we just need to check against `used_diffs` (global) and `current_row` (local).
                            
                            # Wait, `used_diffs` contains ALL diffs from ALL rows.
                            # `current_row` diffs are a subset of `used_diffs`.
                            # So checking against `used_diffs` is sufficient.
                            
                            # Add to next beam
                            new_rows = [r[:] for r in rows]
                            new_rows[r_idx].append(x)
                            
                            new_used_diffs = set(used_diffs)
                            for a in new_rows[r_idx]:
                                if a != x:
                                    new_used_diffs.add(x - a)
                            
                            new_max = max(curr_max, x)
                            
                            if new_max < initial_scope:
                                next_beam.append((new_rows, new_used_diffs, new_max))
                            
                            found = True
                            break
                        x += 1
                    
                    if not found:
                        # This state is a dead end for this row expansion
                        pass
            
            if not next_beam:
                break
            
            # Sort by max_val and keep top beam_width
            next_beam.sort(key=lambda x: x[2])
            current_beam = next_beam[:beam_width]
            
            # Check for complete solution
            for rows, used_diffs, curr_max in current_beam:
                if all(len(r) == row_len for r in rows):
                    if curr_max < initial_scope:
                        return rows, curr_max
            
            if time.time() > start + max_time:
                break
        
        return None, None

    # Phase 1: Get a baseline with greedy
    best_rows = greedy_init(0.15, 30)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local Search Refinement
    local_time = 40.0
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        local_time = max(10, deadline - elapsed - 30)
    
    if local_time > 5:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope
    
    # Phase 3: Beam Search to try to beat the current best
    # We use the current best scope as the target.
    # Beam search is slow, so we limit time.
    
    current_target = best_scope - 1
    search_time = 20.0
    
    if time.time() < deadline - 10:
        # Try to find a solution with scope < best_scope
        # Beam search might be too slow for large scopes, but let's try a short run.
        # We'll run it for a short time.
        
        # Actually, beam search as implemented above expands n_rows branches per state.
        # With beam_width=50, it's 50 * 4 = 200 states per step.
        # 4 rows * 8 elements = 32 steps.
        # Total states processed ~ 200 * 32 = 6400. This is fast.
        
        # Let's try beam search with target best_scope - 1
        if current_target > 112:
            result_rows, result_scope = beam_search(n_rows, k, max_time=10.0, initial_scope=current_target)
            if result_rows and result_scope < best_scope:
                if validate(result_rows):
                    best_rows = result_rows
                    best_scope = result_scope
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 2:
        polish_time = min(10.0, deadline - time.time() - 2)
        if polish_time > 2:
            polished_rows, polished_scope = local_search(best_rows, polish_time)
            if validate(polished_rows):
                best_rows = polished_rows
                best_scope = polished_scope
    
    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
