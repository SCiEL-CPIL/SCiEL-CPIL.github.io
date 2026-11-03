import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def randomized_greedy(n_rows, k, randomness=0.5):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    if random.random() < randomness:
                        # Take smallest
                        row.append(x)
                        used.update(new_diffs)
                    else:
                        # Look ahead
                        candidates = []
                        for dx in range(11):
                            cx = x + dx
                            c_diffs = {cx - a for a in row}
                            if not c_diffs & used:
                                candidates.append((cx, c_diffs))
                            if len(candidates) >= 3:
                                break
                        if candidates:
                            cx, c_diffs = random.choice(candidates)
                            row.append(cx)
                            used.update(c_diffs)
                            x = cx - 1
                        else:
                            row.append(x)
                            used.update(new_diffs)
            rows.append(row)
        return rows

    def local_search(rows, deadline):
        # Precompute diffs per row
        diffs_per_row = [get_diffs(r) for r in rows]
        
        # Union of all diffs
        all_diffs = set()
        for d in diffs_per_row:
            all_diffs.update(d)
            
        best_max = max(max(r) for r in rows)
        current_rows = [list(r) for r in rows]
        current_max = best_max
        
        # Helper to get other diffs
        def get_other_diffs(i):
            other = set()
            for j in range(n_rows):
                if j != i:
                    other.update(diffs_per_row[j])
            return other

        while time.time() < deadline:
            i = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            old_val = current_rows[i][j]
            
            # Remove old diffs from row i
            old_diffs = diffs_per_row[i]
            
            # Calculate new diffs for a candidate value
            # We need to check against other_diffs
            
            other_diffs = get_other_diffs(i)
            
            # Determine candidates
            lower = current_rows[i][j-1] + 1
            upper = current_rows[i][j+1] - 1 if j < k else 1000 # Large upper bound
            
            candidates = set()
            # If old_val is max, try to decrease
            if old_val == best_max:
                candidates.update(range(lower, old_val))
            else:
                candidates.update(range(max(lower, old_val - 5), min(upper, old_val + 5) + 1))
            
            # Random jumps
            for _ in range(5):
                cand = random.randint(lower, upper)
                candidates.add(cand)
            
            # Filter candidates
            valid_candidates = []
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val < lower or new_val > upper:
                    continue
                
                # Calculate new diffs for this row
                # Row is current_rows[i]
                # We replace j with new_val
                new_diffs = set()
                for idx, val in enumerate(current_rows[i]):
                    if idx != j:
                        new_diffs.add(abs(new_val - val))
                        
                # Check internal duplicates
                if len(new_diffs) != k:
                    continue
                
                # Check against other rows
                if new_diffs & other_diffs:
                    continue
                
                valid_candidates.append((new_val, new_diffs))
            
            if valid_candidates:
                # Pick the one that minimizes the new max? Or random?
                # Let's pick the one that minimizes the max of the row
                valid_candidates.sort(key=lambda x: max(x[0], max(current_rows[i][:j]) if j>0 else 0, max(current_rows[i][j+1:]) if j<k else 0))
                # Simpler: just pick the one with smallest new_val if it reduces max, else random
                # Let's just pick the first valid one (smallest new_val) to be greedy
                new_val, new_diffs = valid_candidates[0]
                
                # Apply
                current_rows[i][j] = new_val
                diffs_per_row[i] = new_diffs
                
                # Update max
                new_global_max = max(max(r) for r in current_rows)
                if new_global_max < current_max:
                    current_max = new_global_max
                    best_max = new_global_max

        return current_rows, best_max

    best_solution = randomized_greedy(n_rows, k)
    best_score = max(max(r) for r in best_solution)
    
    improved_solution, improved_score = local_search(best_solution, deadline)
    if improved_score < best_score:
        best_solution = improved_solution
        best_score = improved_score
    
    while time.time() < deadline - 10:
        candidate = randomized_greedy(n_rows, k, randomness=random.uniform(0.3, 0.8))
        cand_score = max(max(r) for r in candidate)
        if cand_score < best_score + 20:
            improved_solution, improved_score = local_search(candidate, deadline)
            if improved_score < best_score:
                best_solution = improved_solution
                best_score = improved_score
            if best_score == 112:
                break

    return best_solution
