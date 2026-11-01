import time
import sys
import random
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115
    
    # 1. Initial Solution via Greedy
    # We construct rows one by one, greedily picking the smallest valid integers.
    # This provides a valid baseline to work with.
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        # Fallback if greedy fails (shouldn't happen for small n,k)
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Precompute difference sets for the initial solution to speed up checks
    # Structure: list of sets of differences for each row
    # Global used diffs set for quick disjoint check
    
    # We will use an Iterated Local Search (ILS) approach.
    # State: list of rows
    # Move: 
    #   1. Local Search: Try to reduce any element in any row to a smaller valid value.
    #   2. Perturbation: Randomly reconstruct a row or swap elements between rows.
    
    current_solution = [r[:] for r in best_solution]
    current_score = best_score
    
    # Parameters for Simulated Annealing / ILS
    # We will run until deadline
    # Temperature starts high, cools down
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline - 5:
        # 1. Local Search Phase: Minimize scope
        improved, new_solution, new_score = _local_search(current_solution, n_rows, k, deadline)
        
        if improved:
            current_solution = new_solution
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_solution = [r[:] for r in current_solution]
                # Reset temperature slightly on improvement? Or just continue.
                # For ILS, we usually accept improvement and perturb.
                # Let's perturb immediately after improvement to move on.
        
        # 2. Perturbation Phase
        # Choose a random row to reconstruct or modify
        if time.time() < deadline - 5:
            perturbed_solution = _perturb(current_solution, n_rows, k, deadline)
            
            if perturbed_solution:
                # Calculate score of perturbed solution
                pert_score = max(max(r) for r in perturbed_solution)
                
                # Acceptance Criterion
                # If better, accept.
                # If worse, accept with probability exp(-delta/T)
                delta = pert_score - current_score
                if delta < 0:
                    current_solution = perturbed_solution
                    current_score = pert_score
                else:
                    # Calculate probability
                    if T > 0.01:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_solution = perturbed_solution
                            current_score = pert_score
                            
                # Cool down
                T *= cooling
                # If T gets too low, reheat occasionally to escape deep local minima
                if T < 0.01 and random.random() < 0.05:
                    T = 5.0
    
    return best_solution

def _greedy_construct(n_rows, k):
    """
    Constructs a valid set of rows using a greedy approach.
    Rows are constructed sequentially. For each row, we pick the smallest
    possible integers that don't conflict with previously used differences.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Search for the next valid integer
            # We search from current_max + 1 upwards
            # To keep it fast, we limit the search range, but for k=5, it's small
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline):
    """
    Attempts to reduce the scope of the solution by replacing elements
    with smaller valid values.
    Returns (improved, new_rows, new_score)
    """
    current_rows = [r[:] for r in rows]
    improved = False
    
    # We iterate multiple times until no improvement is found or time runs out
    # Each pass tries to shrink elements
    while time.time() < deadline - 5:
        pass_improved = False
        
        # Shuffle rows to avoid bias in which row gets optimized first
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for r_idx in row_order:
            if time.time() > deadline - 5:
                break
            
            row = current_rows[r_idx]
            
            # We try to shrink elements from largest to smallest?
            # Or smallest to largest?
            # Shrinking a large element might allow shrinking a smaller one?
            # Actually, shrinking a smaller element (index i) changes diffs with j>i.
            # It's safer to shrink from the end (largest indices) to the start,
            # or just iterate and check validity.
            # Let's iterate from last index down to 1.
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline - 5:
                    break
                
                old_val = row[pos]
                if old_val <= 1:
                    continue
                
                # We want to find the smallest x > row[pos-1] such that
                # {x - row[i] for i in 0..pos-1} U {row[j] - x for j in pos+1..end}
                # is disjoint from (all_diffs - diffs_involving_old_val)
                
                # 1. Identify diffs involving old_val in this row
                diffs_to_remove = set()
                for i in range(pos):
                    diffs_to_remove.add(old_val - row[i])
                for j in range(pos + 1, len(row)):
                    diffs_to_remove.add(row[j] - old_val)
                
                # 2. Identify all other diffs in the system
                # Instead of recalculating all diffs every time (expensive),
                # we can maintain a global set, but here we are in a local search step.
                # For efficiency in this specific function, we'll compute the "forbidden" set
                # which is (all_diffs - diffs_to_remove).
                
                # To avoid O(N^2) recalculations of all_diffs every step,
                # we can precompute the set of all diffs for the current state.
                # But since we are modifying the state, we need to be careful.
                # Let's compute the forbidden set once per row modification attempt.
                
                all_diffs = set()
                for r in current_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            all_diffs.add(r[j] - r[i])
                
                forbidden = all_diffs - diffs_to_remove
                
                # 3. Find smallest valid x
                min_possible = row[pos - 1] + 1
                # We only care about reducing the scope, so we only look for x < old_val
                # If old_val is the global max, reducing it improves score.
                # If old_val is not the global max, reducing it might not improve score,
                # but it might allow OTHER elements to be reduced later.
                # So we always try to find the smallest valid x.
                
                new_val = None
                # Check from min_possible up to old_val - 1
                for x in range(min_possible, old_val):
                    # Check if x creates conflicts within the row
                    valid_in_row = True
                    for i in range(pos):
                        d = x - row[i]
                        if d in forbidden:
                            valid_in_row = False
                            break
                        # Also check if d is already in the row's internal diffs?
                        # No, 'forbidden' includes all diffs except those removed.
                        # The diffs (x - row[i]) are new. They must not be in forbidden.
                        # They also must not duplicate each other (which they won't if row is strictly increasing and x > row[i]).
                        # Wait, do we need to check if (x - row[i]) == (x - row[l])? No, row[i] != row[l].
                        # Do we need to check if (x - row[i]) == (row[j] - x)?
                        # Yes, if i < pos < j, then x - row[i] and row[j] - x are both new diffs.
                        # They must be distinct from each other?
                        # The problem says "all differences between two entries of the same row are distinct".
                        # So within the row, all diffs must be distinct.
                        # If x - row[i] == row[j] - x, that's a collision within the row.
                        # We need to check this.
                        
                        # Let's collect new diffs and check for internal duplicates.
                        pass
                    
                    if not valid_in_row:
                        continue
                        
                    # Re-do the internal check properly
                    new_diffs = set()
                    conflict = False
                    for i in range(pos):
                        d = x - row[i]
                        if d in forbidden:
                            conflict = True
                            break
                        if d in new_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                        
                    if conflict:
                        continue
                        
                    for j in range(pos + 1, len(row)):
                        d = row[j] - x
                        if d in forbidden:
                            conflict = True
                            break
                        if d in new_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                        
                    if conflict:
                        continue
                        
                    # If we reach here, x is valid
                    new_val = x
                    break
                
                if new_val is not None and new_val < old_val:
                    row[pos] = new_val
                    improved = True
                    pass_improved = True
                    # Note: We don't update 'forbidden' immediately, we recalculate in next iterations
                    # This is inefficient but correct.
                    # Optimization: We could maintain the global diff set incrementally.
                    # Given k=5, n=7, the sets are small. Recalculation is O(7*15) = 105 ops.
                    # This is very fast.
                    
            if not pass_improved:
                # No improvement in this pass
                break
                
    final_score = max(max(r) for r in current_rows)
    return improved, current_rows, final_score

def _perturb(rows, n_rows, k, deadline):
    """
    Creates a perturbed version of the solution.
    Strategy: Pick a random row and reconstruct it from scratch,
    ensuring it doesn't conflict with the other rows.
    """
    new_rows = [r[:] for r in rows]
    
    # Pick a random row to reconstruct
    idx = random.randint(0, n_rows - 1)
    
    # Collect diffs from all OTHER rows
    used_diffs = set()
    for i in range(n_rows):
        if i == idx:
            continue
        r = new_rows[i]
        for a in range(len(r)):
            for b in range(a + 1, len(r)):
                used_diffs.add(r[b] - r[a])
    
    # Reconstruct row idx
    # We want it to be compact, so we use a randomized greedy with a bias towards small numbers
    # but allowing some randomness to escape local minima.
    
    row = [0]
    current_max = 0
    
    # We might want to allow the new row to be slightly larger initially?
    # No, we want to minimize.
    
    for _ in range(k):
        if time.time() > deadline - 5:
            return None
            
        # Find candidates
        # We search in a window.
        # To ensure we find a solution, we might need to search far.
        # But we want compact.
        # Let's search up to current_max + 200.
        
        candidates = []
        limit = current_max + 200
        
        for x in range(current_max + 1, limit):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                # Don't take too many candidates to keep it fast
                if len(candidates) >= 20:
                    break
        
        if not candidates:
            # Expand search if needed, or return None (failure)
            # For robustness, let's try a bit further
            for x in range(limit, current_max + 500):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
                    break
            
        if not candidates:
            return None
            
        # Choose a candidate
        # Weighted towards smaller values
        # Weight = 1 / (x - current_max + 1)
        weights = [1.0 / (x - current_max + 1) for x in candidates]
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0
        chosen = candidates[0]
        for x, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                chosen = x
                break
                
        # Update used_diffs with new diffs
        for a in row:
            used_diffs.add(chosen - a)
            
        row.append(chosen)
        current_max = chosen
        
    new_rows[idx] = row
    
    # Verify validity (just in case)
    if not _is_valid(new_rows):
        return None
        
    return new_rows

def _is_valid(rows):
    """
    Checks if the set of rows is valid (all diffs distinct).
    """
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

# Keep the original function name
def construct(n_rows, k):
    # The implementation above is in the main construct function.
    # We just need to ensure the name matches.
    pass

# Re-define construct to use the logic above
def construct(n_rows, k):
    deadline = time.time() + 115
    
    # 1. Initial Solution via Greedy
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # We will use an Iterated Local Search (ILS) approach.
    current_solution = [r[:] for r in best_solution]
    current_score = best_score
    
    # Parameters for Simulated Annealing / ILS
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline - 5:
        # 1. Local Search Phase: Minimize scope
        improved, new_solution, new_score = _local_search(current_solution, n_rows, k, deadline)
        
        if improved:
            current_solution = new_solution
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_solution = [r[:] for r in current_solution]
        
        # 2. Perturbation Phase
        if time.time() < deadline - 5:
            perturbed_solution = _perturb(current_solution, n_rows, k, deadline)
            
            if perturbed_solution:
                pert_score = max(max(r) for r in perturbed_solution)
                
                delta = pert_score - current_score
                if delta < 0:
                    current_solution = perturbed_solution
                    current_score = pert_score
                else:
                    if T > 0.01:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_solution = perturbed_solution
                            current_score = pert_score
                            
                T *= cooling
                if T < 0.01 and random.random() < 0.05:
                    T = 5.0
    
    return best_solution
