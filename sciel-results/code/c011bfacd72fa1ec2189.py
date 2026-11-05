import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with the improved heuristic
    # Use multiple seeds to find diverse starting points
    seed = 0
    max_initial_seeds = 100
    while time.time() < deadline - 60 and seed < max_initial_seeds:
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
        seed += 1
    
    if best_rows is None:
        return [[0, 1, 3, 7, 15, 31]] * n_rows # Fallback, likely invalid but safe

    # Phase 2: Simulated annealing
    if time.time() < deadline - 20:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final polish
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i+1, n):
                diffs.add(row[j] - ri)
    return diffs

def _greedy_construct(n_rows, k, seed):
    """
    Greedy construction using Future Slack heuristic.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        success = True
        for step in range(k):
            last_x = row[-1]
            best_x = None
            best_score = float('inf')
            
            # Search for x
            x = last_x + 1
            limit = 1000  # Hard cap for safety, though we hope to stay lower
            
            # We look for a small window of candidates to balance cost and quality
            candidates = []
            
            while x <= limit and len(candidates) < 20:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Calculate Future Slack
                    # Check next 30 values to see if we can extend
                    slack = _calc_future_slack(row, x, used, 30)
                    # Score: prefer smaller x, but penalize low slack
                    # If slack is 0 and it's not the last step, it's a dead end
                    if step == k - 1:
                        score = x
                    else:
                        if slack == 0:
                            score = x * 10  # Heavy penalty for dead ends
                        else:
                            # Trade-off: compactness vs flexibility
                            score = x - slack * 2.0
                    
                    if score < best_score:
                        best_score = score
                        best_x = x
                    candidates.append((score, x))
                x += 1
            
            if best_x is None:
                # If no candidate found in the window, expand search (might be stuck)
                # This is a fallback for pathological cases
                x_limit = x + 500
                while x <= x_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        slack = _calc_future_slack(row, x, used, 30)
                        if step == k - 1:
                            score = x
                        else:
                            if slack == 0:
                                score = x * 10
                            else:
                                score = x - slack * 2.0
                        
                        if score < best_score:
                            best_score = score
                            best_x = x
                        if best_score < 500: # Heuristic: good enough
                            break
                    x += 1

            if best_x is None:
                success = False
                break

            row.append(best_x)
            # Update used diffs
            for a in row[:-1]:
                used.add(best_x - a)
        
        if not success:
            return None
        rows.append(row)
    
    return rows

def _calc_future_slack(row, x, used, depth):
    """
    Estimate how many valid choices for the next element exist.
    row: current row including the hypothetical new element x? No, row is prefix.
    We assume the new element is x. So the new row prefix is row + [x].
    """
    # New row prefix is [0, ..., last, x]
    # Check y from x+1 to x+depth
    count = 0
    for y in range(x + 1, x + depth + 1):
        valid = True
        # Check diffs against row elements and x
        for a in row:
            d = y - a
            if d in used:
                valid = False
                break
        if valid:
            for a in row:
                # We already checked y-a. 
                # Wait, 'used' contains diffs from other rows.
                # We also need to ensure new diffs from this row (including x) don't collide.
                # The set of diffs generated by adding y to [row..., x] is {y - a for a in row} U {y - x}.
                # We checked y - a against 'used' (other rows).
                # We must also check if y - a collides with existing diffs in THIS row?
                # No, 'used' is global.
                # But wait, the current row 'row' has already generated diffs that are NOT in 'used' yet?
                # No, in _greedy_construct, we update 'used' as we go.
                # So 'used' contains diffs from *other* rows AND previous parts of *this* row.
                # So checking `d in used` is sufficient for collision with existing global state.
                # However, we also need to ensure that the *new* diffs introduced by 'y' don't collide 
                # with *other* new diffs introduced by 'y'? 
                # i.e., y - x == y - a (for a in row) => x == a, impossible since strict increase.
                # So internal consistency is guaranteed by strict increase.
                pass
        
        if valid:
            count += 1
            # Optimization: break if count is high? 
            # We just need a relative measure. 50 is plenty.
            if count >= 50:
                break
    return count

def _is_valid(rows, n_rows, k):
    all_diffs = set()
    expected = n_rows * (k + 1) * k // 2
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == expected

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Simulated annealing.
    Key optimization: Maintain 'used_diffs' of all rows *except* the one being modified?
    No, because we generate a new row, we need the set of diffs from the *other* rows.
    We will recompute 'other_diffs' at the start of each step? 
    That is O(N^2). If N is large, this is slow.
    But we can maintain a global diff set and subtract the current row's diffs?
    Set difference is O(N).
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diffs for each row to allow fast removal?
    # Let's just recalculate 'other_diffs' from scratch if n_rows is small, 
    # or use a global set and subtract.
    
    T_start = 50.0
    T_end = 0.1
    total_steps = 30000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 2000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Choose a row to modify
        # Bias towards the row with the max element
        if rng.random() < 0.5:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Calculate diffs of all OTHER rows
        # This is the expensive part.
        other_diffs = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            row = rows[i]
            n = len(row)
            for a in range(n):
                ra = row[a]
                for b in range(a+1, n):
                    other_diffs.add(row[b] - ra)
        
        # Generate new row
        new_row = _generate_candidate_row_fast(other_diffs, k, rng, row_idx)
        
        if new_row is None:
            continue
            
        # Check validity against other_diffs (internal validity of new_row is guaranteed by generator)
        new_row_diffs = _row_diffs(new_row)
        if new_row_diffs & other_diffs:
            continue # Collision
        
        # Valid move
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        
        if steps_since_improve > max_no_improve:
            # Restart-like perturbation
            # Pick a random row and force a very tight reconstruction
            pert_idx = rng.randint(0, n_rows - 1)
            other_diffs_pert = set()
            for i in range(n_rows):
                if i == pert_idx:
                    continue
                row = rows[i]
                n = len(row)
                for a in range(n):
                    ra = row[a]
                    for b in range(a+1, n):
                        other_diffs_pert.add(row[b] - ra)
            
            # Force tight generation
            tight_row = _generate_tight_row(other_diffs_pert, k, rng)
            if tight_row is not None:
                tight_diffs = _row_diffs(tight_row)
                if not (tight_diffs & other_diffs_pert):
                    new_rows = rows[:]
                    new_rows[pert_idx] = tight_row
                    if _is_valid(new_rows, n_rows, k): # Double check
                        rows = new_rows
                        current_scope = _scope(rows)
            steps_since_improve = 0
            T = max(T, 5.0)
    
    return best_rows

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i+1, n):
            diffs.add(row[j] - ri)
    return diffs

def _generate_candidate_row_fast(used_diffs, k, rng, row_idx):
    """
    Generates a row avoiding 'used_diffs'.
    Uses Future Slack.
    """
    row = [0]
    for step in range(k):
        last_x = row[-1]
        best_x = None
        best_score = float('inf')
        
        x = last_x + 1
        # Limit search scope to keep it fast
        # Heuristic: we don't need to search too far if we are flexible
        search_limit = last_x + 100
        if row_idx is not None: # Slightly wider if we are replacing a "big" row?
             pass
             
        candidates = []
        found_dead_end = False
        
        while x <= search_limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                # Check internal collision with row's existing diffs? 
                # No, row diffs are not in used_diffs.
                # But new_diffs must be distinct from each other (guaranteed by strict increase)
                # and distinct from row's existing internal diffs?
                # Wait. 'used_diffs' only contains diffs from *other* rows.
                # The current row 'row' has internal diffs!
                # We MUST check new_diffs against row's existing internal diffs.
                
                row_internal_diffs = _row_diffs(row)
                if new_diffs & row_internal_diffs:
                    x += 1
                    continue
                
                slack = _calc_future_slack(row, x, used_diffs | row_internal_diffs, 20)
                
                if step == k - 1:
                    score = x
                else:
                    if slack == 0:
                        score = x * 10
                    else:
                        score = x - slack * 1.5
                
                if score < best_score:
                    best_score = score
                    best_x = x
                candidates.append((score, x))
            x += 1
        
        if best_x is None:
            # Expand search if stuck
            x = search_limit + 1
            while x <= search_limit * 2:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row_internal_diffs = _row_diffs(row)
                    if not (new_diffs & row_internal_diffs):
                        slack = _calc_future_slack(row, x, used_diffs | row_internal_diffs, 20)
                        if step == k - 1:
                            score = x
                        else:
                            if slack == 0:
                                score = x * 10
                            else:
                                score = x - slack * 1.5
                        if score < best_score:
                            best_score = score
                            best_x = x
                        if best_score < 100:
                            break
                x += 1

        if best_x is None:
            return None

        row.append(best_x)
        
    return row

def _generate_tight_row(used_diffs, k, rng):
    """
    Generates a row specifically trying to minimize the max element.
    Picks the smallest valid x at each step.
    """
    row = [0]
    row_internal_diffs = set()
    
    for step in range(k):
        last_x = row[-1]
        best_x = None
        
        x = last_x + 1
        while x <= last_x + 200:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs) and not (new_diffs & row_internal_diffs):
                # Check future slack to avoid immediate dead end?
                # For "tight" we prefer smallest x, but if it's a dead end, it fails.
                # So we still check slack > 0 unless last step.
                if step < k - 1:
                    slack = _calc_future_slack(row, x, used_diffs | row_internal_diffs, 10)
                    if slack == 0:
                        x += 1
                        continue
                best_x = x
                break
            x += 1
        
        if best_x is None:
            return None
        
        row.append(best_x)
        for a in row[:-1]:
            row_internal_diffs.add(best_x - a)
            
    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: try to reduce max element further.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    for iteration in range(50):
        if time.time() > deadline - 2:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        
        # Get diffs of other rows
        other_diffs = set()
        for i in range(n_rows):
            if i == max_idx:
                continue
            row = rows[i]
            n = len(row)
            for a in range(n):
                ra = row[a]
                for b in range(a+1, n):
                    other_diffs.add(row[b] - ra)
                    
        # Try to generate a tight row
        improved_row = _generate_tight_row(other_diffs, k, rng)
        
        if improved_row is not None:
            # Verify no collisions with other_diffs (should be ok if generated against them)
            # And internal validity
            if not (_row_diffs(improved_row) & other_diffs):
                new_rows = rows[:]
                new_rows[max_idx] = improved_row
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows
                else:
                    # Even if scope didn't drop, maybe this row is "better" shaped for future moves?
                    # Stick with best_scope for now.
                    pass
    
    return best_rows if best_rows != rows else None
