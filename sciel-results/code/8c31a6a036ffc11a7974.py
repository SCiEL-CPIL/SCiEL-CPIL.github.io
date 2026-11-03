import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _generate_candidates(k, max_scope, deadline):
    """
    Generate all valid rows of length k+1 starting at 0, with max element <= max_scope.
    Returns a list of (row, diff_set, max_val).
    """
    n = k + 1
    candidates = []
    
    def backtrack(pos, current_row, current_diffs, max_val_so_far):
        if time.time() > deadline:
            return
        if pos == n:
            # Valid row found
            # Check if all diffs are unique (implied by construction if we check against current_diffs)
            # current_diffs contains all pairwise diffs. If any were duplicate, we would have pruned.
            candidates.append((list(current_row), frozenset(current_diffs), max_val_so_far))
            return
        
        prev = current_row[-1]
        # The next element must be > prev
        # And max_val_so_far must be <= max_scope
        # Also, if this is the last element, it must be <= max_scope
        # If not last, it can be up to max_scope, but we should ensure we can fit the rest.
        
        # Heuristic: if we are at pos, we have n - pos - 1 elements left.
        # The minimum increment is 1. So current_row[-1] + (n - pos - 1) <= max_scope
        min_next = prev + 1
        max_next = max_scope
        
        # Pruning: if we pick max_next, can we fit the rest?
        # Not strictly necessary if we just iterate up to max_scope, but good for speed.
        
        for val in range(min_next, max_next + 1):
            # Check conflicts with existing elements
            conflict = False
            new_diffs_to_add = []
            for i in range(pos):
                d = val - current_row[i]
                if d in current_diffs:
                    conflict = True
                    break
                new_diffs_to_add.append(d)
            
            if conflict:
                continue
            
            # Also, if we are not at the last position, we must ensure that the new diffs
            # don't prevent us from placing future elements? 
            # Actually, the check `d in current_diffs` ensures internal uniqueness.
            # We don't need to check future conflicts here, just current.
            
            current_row.append(val)
            current_diffs.update(new_diffs_to_add)
            
            # Recurse
            backtrack(pos + 1, current_row, current_diffs, max(max_val_so_far, val))
            
            # Backtrack
            current_row.pop()
            for d in new_diffs_to_add:
                current_diffs.discard(d)
                
    backtrack(1, [0], set(), 0)
    return candidates

def _greedy_mis(candidates, n_rows, deadline):
    """
    Greedy Maximal Independent Set on the candidates.
    candidates: list of (row, diff_set, max_val)
    Returns a list of rows or None.
    """
    # Sort by max_val, then maybe by "density" or just stable sort
    # We want small scopes first.
    candidates.sort(key=lambda x: x[2])
    
    used_diffs = set()
    selected_rows = []
    
    for row, diff_set, max_val in candidates:
        if time.time() > deadline:
            break
        if len(selected_rows) == n_rows:
            break
        
        # Check if disjoint
        if diff_set.isdisjoint(used_diffs):
            selected_rows.append(row)
            used_diffs.update(diff_set)
            
    if len(selected_rows) == n_rows:
        return selected_rows
    return None

def _sa_optimize(rows, k, deadline, rng, max_iters=50000):
    """
    Simulated Annealing to minimize the scope.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Track all differences for quick collision checking
    all_diffs = []
    for r in current:
        all_diffs.extend(_diffs_for_row(r))
    diff_count = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in diff_count.values())
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    T = 50.0
    cooling = 0.999
    min_T = 0.05
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Choose a random row and position to modify
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 100
        
        # Determine move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Scope compression: try to reduce a value
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
            if new_val >= old_val:
                new_val = old_val - 1
            if new_val <= prev_val:
                continue
        elif move_type < 0.7:
            # Small perturbation
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 15))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        
        # Compute the difference changes
        old_diffs = _diffs_for_row(current[i])
        
        # Update the row temporarily
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        
        # Check validity of new row
        valid_row = len(set(new_diffs)) == len(new_diffs) and all(d > 0 for d in new_diffs)
        
        if not valid_row:
            current[i][j] = old_val
            continue
        
        # Compute scope change
        new_scope = max(current_scope, max(new_row))
        
        # Compute collision change
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        
        new_collisions = sum(max(0, c - 1) for c in diff_count.values())
        
        # Cost function
        scope_penalty = 5000 if current_scope > 120 else 3000 if current_scope > 110 else 2000
        old_cost = current_scope + scope_penalty * collisions
        new_cost = new_scope + scope_penalty * new_collisions
        
        # Accept if better or with probability based on temperature
        if new_cost < old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            if delta_cost < 0:
                accept = True
            else:
                accept_prob = math.exp(-delta_cost / max(T, min_T))
                accept = rng.random() < accept_prob
        
        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin
    rng = random.Random(123)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Structural Search (MIS on Difference Sets)
    # Start from a lower bound. Counting bound is roughly n_rows * C(k+1, 2) / 2 ?
    # Actually, just start from a reasonable estimate.
    # For k=6, C(7,2)=21. 7 rows need 147 diffs. If packed tightly, scope ~ 147?
    # But diffs are not just 1..M, they are specific values.
    # Let's try M from 105 to 130.
    
    # To avoid generating too many candidates, we limit the generation time per M.
    # If generation takes too long, we skip to next M or use SA.
    
    start_m = 105
    end_m = 135
    
    # We will try a few random orders of candidates to escape local optima of the greedy MIS
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
            
        # Shuffle the random seed for candidate ordering? 
        # Actually, we sort by scope. To break ties, we can randomize the order of candidates with the same scope.
        # Or we can just run the greedy multiple times with different random subsets?
        # Let's stick to sorted by scope, but if we fail, we increase M.
        
        found_in_phase = False
        for m in range(start_m, end_m + 1):
            if time.time() > deadline - 20:
                break
                
            gen_deadline = time.time() + 5.0 # Limit generation time per M
            candidates = _generate_candidates(k, m, gen_deadline)
            
            if not candidates:
                continue
                
            # If candidates list is huge, it might be slow to sort and iterate.
            # But for m < 130 and k=6, it should be manageable.
            
            # Run Greedy MIS
            rows = _greedy_mis(candidates, n_rows, deadline - 10)
            
            if rows is not None:
                s = _scope(rows)
                if s < best_score:
                    best_score = s
                    best_result = rows
                    found_in_phase = True
                    # If we found a solution, we can stop increasing M for this attempt
                    # unless we want to check if a smaller M was possible (we iterate M ascending, so first found is min for this candidate set)
                    break
            
            # If we didn't find a solution, we might need to try a different candidate set?
            # No, if we generated ALL candidates up to M, and greedy failed, it doesn't mean no solution exists.
            # Greedy MIS is not optimal. 
            # However, if we increase M, we add more candidates, which might allow a solution.
            
        if found_in_phase:
            # We found a solution. Let's try to polish it with SA.
            # But first, let's see if we can find a better one with a different randomization?
            # For now, break to save time for SA.
            break
            
    # Phase 2: Simulated Annealing Polish
    if best_result is not None:
        if time.time() < deadline - 10:
            remaining = deadline - time.time()
            sa_deadline = time.time() + remaining * 0.8
            s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=int(remaining * 20000))
            if _valid(rows, k) and s < best_score:
                best_score = s
                best_result = rows
    else:
        # Fallback: Generate a valid set using a simple greedy on values (like the original seed)
        # This ensures we return *something* valid even if MIS fails or times out.
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        
        if time.time() < deadline - 10:
            remaining = deadline - time.time()
            s, rows = _sa_optimize(rows, k, time.time() + remaining * 0.9, rng, max_iters=int(remaining * 20000))
            if _valid(rows, k):
                best_score = s
                best_result = rows
            else:
                best_result = rows # Return the unpolished one if SA failed? No, SA returns best found.
        
    return best_result
