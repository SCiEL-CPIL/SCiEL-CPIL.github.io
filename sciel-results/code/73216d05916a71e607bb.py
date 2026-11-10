import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    # Use 115s of the 120s limit to be safe, but allow for quick exit if possible
    deadline = start_time + 115.0
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound: M*(M+1)/2 >= total_diffs
    # M >= (-1 + sqrt(1 + 8*total_diffs)) / 2
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    best_solution = None
    best_score = float('inf')
    
    # Helper to check time
    def time_up():
        return time.time() > deadline
    
    # Fast validation of a complete set of rows
    def validate(rows):
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # MRV Backtracking Search
    # State:
    # rows: list of lists. rows[r] contains the elements placed in row r so far.
    #       Initially [[0] for _ in range(n_rows)]
    # used_diffs: set of all differences used so far.
    # 
    # The "variables" are the positions (r, idx) where idx is 1..k for each row r.
    # We need to assign a value x > rows[r][-1] to each variable.
    # 
    # MRV Heuristic:
    # For each unassigned variable (r, idx), the "domain" is effectively x > current_max.
    # However, the "constraint" is that {x - a for a in rows[r]} must not intersect used_diffs.
    # The "scope" or "feasibility" can be estimated by how many small x are blocked.
    # A simpler MRV proxy: Pick the row that is "most constrained" in terms of 
    # the smallest available x for its next element.
    # Why? If a row is stuck with a large gap, it might force other rows to be larger.
    # Actually, a better MRV for packing is often "pick the variable with the smallest domain".
    # Here, the domain is infinite, but the *cost* of placing x is x.
    # So, we prioritize placing the element that has the *smallest* valid x available.
    # This is effectively "Greedy by MRV" where MRV is defined by the smallest feasible value.
    # 
    # Wait, standard MRV picks the variable with the fewest legal values.
    # Here, every x > max is a candidate. The "fewest legal values" isn't directly applicable.
    # However, we can define "constrainedness" as the size of the smallest valid x.
    # If row A's next smallest valid x is 10, and row B's is 50, Row A is "more constrained" 
    # in the sense that it's blocking more of the low-number space? No, Row B is blocking nothing low.
    # Actually, if we pick the row with the *smallest* next valid x, we are packing tightly.
    # If we pick the row with the *largest* next valid x, we are being lazy.
    # 
    # Let's try: At each step, identify all rows that need more elements.
    # For each such row, find the smallest valid x.
    # Select the row with the smallest such x.
    # This is a "Greediest" approach but with global view of which row *can* take a small number.
    # 
    # To add randomness (for restarts), we can:
    # 1. Randomize the initial order of rows? (Doesn't matter much if we pick min-x each time).
    # 2. With probability p, pick a row that is NOT the one with the absolute smallest x, 
    #    but one of the top few smallest x candidates.
    # 3. Or, when selecting x for the chosen row, pick from the top few valid x's instead of just the smallest.
    
    def backtrack_search(random_seed, max_depth=10000):
        nonlocal best_solution, best_score, deadline
        
        random.seed(random_seed)
        
        # Initialize rows with 0
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # Track the current max value to estimate score
        current_max = 0
        
        # We will build the solution step by step.
        # Total steps = n_rows * k
        total_steps = n_rows * k
        steps_done = 0
        
        # For MRV, we need to find the "best" next move.
        # A move is defined by (row_idx, value).
        # We evaluate all rows that are not full.
        
        while steps_done < total_steps:
            if time.time() > deadline:
                return None
            
            # Find candidates for each incomplete row
            candidates = []
            
            for r in range(n_rows):
                if len(rows[r]) == k + 1:
                    continue
                
                current_row = rows[r]
                last_val = current_row[-1]
                
                # Find the smallest few valid x
                # We need x > last_val such that {x - a for a in current_row} are unused.
                # Note: 0 is in current_row, so x itself must be unused.
                
                valid_xs = []
                x = last_val + 1
                # Limit search window for candidates to keep it fast
                # If the gap is huge, we might need to look further, but usually it's small.
                # Let's look up to 20 candidates or until we find 5 valid ones.
                checks = 0
                while len(valid_xs) < 5 and checks < 100:
                    # Check conflicts
                    conflict = False
                    # Optimization: if x is in used_diffs, conflict (since 0 is in row)
                    if x in used_diffs:
                        x += 1
                        checks += 1
                        continue
                        
                    # Check other differences
                    # We can skip the check for x-0=x since we just checked x
                    for a in current_row[1:]: # Skip 0
                        if (x - a) in used_diffs:
                            conflict = True
                            break
                            
                    if not conflict:
                        valid_xs.append(x)
                        
                    x += 1
                    checks += 1
                
                if not valid_xs:
                    # This row has no valid x in the immediate vicinity?
                    # This shouldn't happen unless we are very constrained.
                    # If it happens, this row is "stuck" or we need to look much further.
                    # For MRV, if a row has NO candidates in the window, it's the most constrained?
                    # Or it's a dead end?
                    # In this problem, there is always a valid x (just make it huge).
                    # So we assume valid_xs is not empty if we search far enough.
                    # If we didn't find any, we force the smallest possible x that works, 
                    # but that requires a full scan. Let's just assume we found some.
                    # If valid_xs is empty, we return None to trigger restart.
                    return None
                    
                # The "constrainedness" metric:
                # We want to pick the row where the smallest valid x is smallest.
                # This packs the numbers tightly.
                min_x = valid_xs[0]
                
                # Add to candidates list
                # We store (min_x, row_idx, valid_xs)
                candidates.append((min_x, r, valid_xs))
            
            if not candidates:
                # All rows full? Should be handled by steps_done check.
                break
                
            # Sort candidates by min_x (smallest first)
            candidates.sort(key=lambda item: item[0])
            
            # Select a candidate
            # Strategy: With high probability, pick the one with the smallest min_x.
            # With low probability, pick a random one from the top 3 to allow diversification.
            if random.random() < 0.85:
                chosen_idx = 0
            else:
                # Pick from top min(3, len(candidates))
                pick_from = min(3, len(candidates))
                chosen_idx = random.randint(0, pick_from - 1)
                
            min_x, r, valid_xs = candidates[chosen_idx]
            
            # Now, which x to pick from valid_xs?
            # Again, biased towards smallest.
            if random.random() < 0.8:
                chosen_x = valid_xs[0]
            else:
                # Pick from top 3 valid x's for this row
                pick_from = min(3, len(valid_xs))
                chosen_x = valid_xs[random.randint(0, pick_from - 1)]
            
            # Update state
            rows[r].append(chosen_x)
            # Update used_diffs
            for a in rows[r]:
                if a == 0:
                    continue
                # Actually, we only add NEW differences.
                # The new element is chosen_x.
                # Differences are chosen_x - a for all a in rows[r] BEFORE adding chosen_x.
                pass
            
            # Let's do the update properly
            # We appended chosen_x to rows[r].
            # The new differences are chosen_x - a for all a in rows[r][:-1]
            for a in rows[r][:-1]:
                d = chosen_x - a
                used_diffs.add(d)
                
            if chosen_x > current_max:
                current_max = chosen_x
                
            steps_done += 1
            
            # Early termination check:
            # If current_max is already worse than best_score, we can stop this branch.
            # But since we are building up, current_max only increases.
            # If current_max >= best_score, no improvement possible.
            if current_max >= best_score:
                return None

        # Construct complete
        # Verify
        if not validate(rows):
            return None
            
        score = current_max
        if score < best_score:
            best_score = score
            best_solution = [row[:] for row in rows]
            
        return score

    # Run multiple restarts with different seeds
    # We need to balance the number of restarts with the depth of search.
    # Since this is a greedy-like backtracking (no actual backtracking on failure, just restart),
    # each run is O(total_steps * n_rows * k) roughly.
    # For n_rows=4, k=7, steps=28. Very fast.
    # We can run many restarts.
    
    seed = 42
    while time.time() < deadline:
        result = backtrack_search(seed)
        if result is not None and result < best_score:
            # Found improvement
            # We could stop if we hit lower bound
            if best_score == lower_bound:
                break
        seed += 1
        # Occasionally, if we are stuck, we might want to try a different randomization strategy?
        # For now, fixed strategy with new seed.
        
    if best_solution is None:
        # Fallback to a simple greedy if backtracking failed (shouldn't happen)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        best_solution = rows
        best_score = max(max(row) for row in rows)

    return best_solution
