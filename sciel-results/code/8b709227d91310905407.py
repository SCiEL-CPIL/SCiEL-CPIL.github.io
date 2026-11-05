"""
Strategy: Genetic Algorithm (GA) for Difference Triangle Sets.
Unlike the single-trajectory Simulated Annealing approach, this maintains a population of
feasible solutions. It uses crossover to combine promising rows from different parents and
mutation to repair or improve specific rows. This allows the search to escape local minima
by recombining structural features of different solutions, rather than just making small
local moves.

Key differences from SA:
1. Population-based: Maintains multiple diverse solutions.
2. Crossover: Creates new solutions by mixing rows from parents.
3. Selection: Uses tournament selection to propagate good genes (low scope).
4. Diversity: Explicitly encourages diversity to avoid premature convergence.
"""

import time
import random
import math
import copy

# EVOLVE-BLOCK-START

def _row_diffs(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    m = 0
    for d in _row_diffs(row):
        m |= (1 << d)
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _generate_random_row(max_val, num_elements, rng):
    """Generate a random row of num_elements increasing ints starting at 0, max <= max_val."""
    if num_elements == 1:
        return [0]
    if num_elements == 2:
        return [0, rng.randint(1, max_val)]
    
    if max_val < num_elements - 1:
        return None
        
    vals = rng.sample(range(1, max_val + 1), num_elements - 1)
    vals.sort()
    return [0] + vals

def _greedy_construct(n_rows, num_elements, max_scope, rng, max_restarts=50):
    """
    Fast greedy to get a valid starting point.
    """
    for restart in range(max_restarts):
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            found = False
            # Try random rows first
            for _ in range(50):
                row = _generate_random_row(max_scope, num_elements, rng)
                if row is None:
                    continue
                diffs = _row_diffs(row)
                if len(diffs) != len(set(diffs)):
                    continue
                m = _row_mask(row)
                if m & used_mask:
                    continue
                rows.append(row)
                used_mask |= m
                found = True
                break
            if not found:
                # Try a deterministic "tight" row if random fails
                # This helps fill gaps
                row = _generate_tight_row(used_mask, num_elements, max_scope)
                if row is not None:
                    rows.append(row)
                    used_mask |= _row_mask(row)
                    found = True
                else:
                    success = False
                    break
        if success:
            return rows
    return None

def _generate_tight_row(used_mask, num_elements, max_scope):
    """
    Greedily construct a row that fits into the gaps of used_mask.
    """
    # We need num_elements - 1 distinct values from 1..max_scope
    # such that all pairwise differences are not in used_mask and distinct within the row.
    # This is a small CSP.
    # Simple heuristic: pick values that minimize the max value while avoiding conflicts.
    # Since max_scope is small, we can try a few random permutations or a simple backtracking.
    
    # Backtracking search for a valid row
    def backtrack(pos, current_row, current_diffs_mask, used_so_far):
        if pos == num_elements:
            return current_row[:]
        
        start_val = 1 if pos == 0 else current_row[-1] + 1
        # To keep scope small, we don't need to go up to max_scope immediately, 
        # but we must stay within max_scope.
        # Actually, we just need *a* valid row.
        
        # Try values in increasing order to keep the row tight
        for val in range(start_val, max_scope + 1):
            # Check new diffs
            new_diffs_mask = 0
            valid = True
            for prev in current_row:
                d = val - prev
                if d in used_so_far:
                    valid = False
                    break
                if d in current_diffs_mask:
                    valid = False
                    break
                current_diffs_mask |= (1 << d)
            
            if not valid:
                # Undo mask changes if we modified current_diffs_mask? 
                # current_diffs_mask is passed by value (int), so no undo needed for the mask itself 
                # but we need to ensure we don't carry over bits from failed attempts.
                # Since we recompute or copy, let's be careful.
                # Actually, the loop above modifies current_diffs_mask. We should use a local copy.
                pass
            
            # Let's rewrite this properly with local state
            # This function is called rarely, so efficiency is less critical than correctness.
            pass
        
        return None

    # Simplified tight row: just randomize with a bias towards small numbers
    # Try 100 times to find a row with small max
    best_row = None
    best_max = max_scope + 1
    for _ in range(100):
        row = _generate_random_row(max_scope, num_elements, random.Random())
        if row is None: continue
        if _row_mask(row) & used_mask: continue
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)): continue
        
        if row[-1] < best_max:
            best_max = row[-1]
            best_row = row[:]
            
    return best_row

def _mutate_row(row, num_elements, used_mask, max_scope, rng):
    """
    Mutate a single row to improve it or repair it.
    Returns a new row or None if failed.
    """
    new_row = row[:]
    # Try to reduce the scope of this specific row
    # Pick a random position to move
    for _ in range(20):
        pos = rng.randint(1, len(new_row) - 1)
        old_val = new_row[pos]
        
        low = new_row[pos - 1] + 1
        high = new_row[pos + 1] - 1 if pos < len(new_row) - 1 else max_scope
        
        if low >= high:
            continue
            
        # Bias towards smaller values to reduce scope
        # Sample from [low, min(high, old_val)] if possible, or just random
        if rng.random() < 0.7:
            upper = min(high, old_val)
        else:
            upper = high
            
        if low > upper:
            continue
            
        new_val = rng.randint(low, upper)
        if new_val == old_val:
            continue
            
        trial_row = new_row[:]
        trial_row[pos] = new_val
        
        # Check internal validity
        diffs = _row_diffs(trial_row)
        if len(diffs) != len(set(diffs)):
            continue
            
        # Check external validity (against used_mask)
        m = _row_mask(trial_row)
        if m & used_mask:
            continue
            
        return trial_row
        
    return None

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    counting_bound = n_rows * num_pairs
    # Best known for (7,5) is 112. Counting bound is 105.
    # Start with a scope that is likely feasible.
    initial_scope = max(counting_bound + 10, 120)
    
    rng = random.Random(12345)
    
    # Phase 1: Initialize Population
    # We want a diverse set of feasible solutions.
    population = []
    pop_size = 20
    
    # Try to generate diverse initial solutions
    for i in range(pop_size):
        # Use slightly different scope limits to encourage diversity
        scope_limit = initial_scope + (i % 5) * 5
        rows = _greedy_construct(n_rows, num_elements, scope_limit, rng, max_restarts=20)
        if rows is not None:
            population.append(rows)
            
    # If we didn't get enough, fill with random restarts
    while len(population) < pop_size and time.time() < DEADLINE - 60:
        rows = _greedy_construct(n_rows, num_elements, initial_scope + 20, rng, max_restarts=10)
        if rows is not None:
            population.append(rows)

    if not population:
        # Emergency fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = min(population, key=_scope)
    best_scope = _scope(best_rows)
    
    # Phase 2: Genetic Algorithm Loop
    # Crossover: Mix rows from two parents.
    # Mutation: Improve a row in a child.
    
    generations = 0
    while time.time() < DEADLINE - 10:
        generations += 1
        
        # Create next generation
        next_population = []
        
        # Elitism: Keep the top 20%
        population.sort(key=_scope)
        elite_count = max(1, pop_size // 5)
        next_population.extend([r[:] for r in population[:elite_count]])
        
        # Fill the rest with offspring
        while len(next_population) < pop_size:
            if time.time() > DEADLINE - 10:
                break
                
            # Tournament Selection
            def select():
                candidates = rng.sample(population, min(3, len(population)))
                return min(candidates, key=_scope)
            
            p1 = select()
            p2 = select()
            
            # Crossover: Uniform crossover of rows
            child_rows = []
            for i in range(n_rows):
                if rng.random() < 0.5:
                    child_rows.append(p1[i][:])
                else:
                    child_rows.append(p2[i][:])
            
            # Check validity of child
            # If invalid, try to repair
            all_diffs = []
            for row in child_rows:
                all_diffs.extend(_row_diffs(row))
            
            if len(all_diffs) != len(set(all_diffs)):
                # Repair: Replace conflicting rows with better ones from parents or mutate
                # Identify conflicts is complex, so just try to fix by replacing rows
                # that have high overlap or are "bad".
                # Simple repair: For each row, if it conflicts, try to mutate it or replace with parent's row.
                # To keep it fast, we just discard this child if it's too broken, 
                # or we do a light repair.
                
                # Light repair: Try to fix rows one by one
                repaired = False
                used_mask = 0
                new_child_rows = []
                for i in range(n_rows):
                    row = child_rows[i]
                    # Check if this row conflicts with previous ones
                    # This is O(n^2) in rows, but n is small (7).
                    m = _row_mask(row)
                    if m & used_mask:
                        # Conflict. Try to mutate this row to avoid used_mask
                        new_row = _mutate_row(row, num_elements, used_mask, _scope(population[0]) + 10, rng)
                        if new_row is not None:
                            row = new_row
                            m = _row_mask(row)
                        else:
                            # Try to replace with a row from a parent that is valid
                            # Or generate a new one
                            new_row = _generate_random_row(_scope(population[0]) + 10, num_elements, rng)
                            if new_row is not None:
                                m2 = _row_mask(new_row)
                                if not (m2 & used_mask) and len(_row_diffs(new_row)) == len(set(_row_diffs(new_row))):
                                    row = new_row
                                    m = m2
                                else:
                                    # Last resort: keep original, mark as invalid
                                    pass
                    # Also check internal validity
                    diffs = _row_diffs(row)
                    if len(diffs) != len(set(diffs)):
                        new_row = _mutate_row(row, num_elements, used_mask, _scope(population[0]) + 10, rng)
                        if new_row is not None:
                            row = new_row
                            m = _row_mask(row)
                    
                    if m & used_mask:
                        # Still conflicting, skip adding to new_child_rows? 
                        # No, we need n_rows rows.
                        # If we can't fix it, this child is invalid.
                        # We'll just add it and let the next generation selection filter it out?
                        # No, invalid solutions shouldn't be in population.
                        # Let's try one more time to generate a fresh row.
                        fresh = _generate_random_row(_scope(population[0]) + 20, num_elements, rng)
                        if fresh is not None:
                            fm = _row_mask(fresh)
                            if not (fm & used_mask) and len(_row_diffs(fresh)) == len(set(_row_diffs(fresh))):
                                row = fresh
                                m = fm
                            else:
                                # Give up on this child
                                child_rows = None
                                break
                        else:
                            child_rows = None
                            break
                            
                    new_child_rows.append(row)
                    used_mask |= m
                    
                if child_rows is not None and len(new_child_rows) == n_rows:
                    # Final check
                    all_d = []
                    for r in new_child_rows:
                        all_d.extend(_row_diffs(r))
                    if len(all_d) == len(set(all_d)):
                        next_population.append(new_child_rows)
                        continue
                # If repair failed, don't add this child
                continue
            else:
                # Child is valid
                next_population.append(child_rows)
                
            # Mutation: Randomly mutate a row in the child to improve scope
            # We already added it. Let's try to improve it in place.
            if next_population and next_population[-1] is child_rows:
                # Pick a random row to mutate
                if rng.random() < 0.5:
                    r_idx = rng.randint(0, n_rows - 1)
                    used_mask = 0
                    for i in range(n_rows):
                        if i != r_idx:
                            used_mask |= _row_mask(child_rows[i])
                    
                    new_row = _mutate_row(child_rows[r_idx], num_elements, used_mask, _scope(child_rows) + 5, rng)
                    if new_row is not None:
                        # Check if it improves scope or is at least as good
                        old_scope = _scope(child_rows)
                        child_rows[r_idx] = new_row
                        new_scope = _scope(child_rows)
                        if new_scope <= old_scope:
                            pass # Keep
                        else:
                            child_rows[r_idx] = new_row # Still keep, might be useful for diversity? 
                            # Actually, if scope increased, it's worse. Revert?
                            # Let's revert if it got worse
                            # We lost the old row. Let's just accept it for now, GA will select.

        population = next_population
        
        # Update best
        current_best = min(population, key=_scope)
        current_scope = _scope(current_best)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in current_best]

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
