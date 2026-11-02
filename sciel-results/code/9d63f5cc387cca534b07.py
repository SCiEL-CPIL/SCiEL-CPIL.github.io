import time
import random
import copy
import math
from scipy.optimize import linear_sum_assignment

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    dset = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            dset.add(row[j] - row[i])
    return dset

def _greedy_row(used, k, max_x=None):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    while len(row) < k + 1:
        base = row[-1]
        found = False
        for x in range(base + 1, (max_x if max_x is not None else 500) + 1):
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, rng, best_score, deadline):
    """
    Randomized greedy construction with multiple strategies.
    """
    used = set()
    rows = []
    success = True
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            candidates = []
            
            # Try a range of values
            search_hi = min(base + 100, best_score + 1)
            for x in range(base + 1, search_hi + 1):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    candidates.append(x)
            
            if not candidates:
                # Expand search
                for x in range(base + 1, best_score + 1):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                        break
                if not candidates:
                    success = False
                    break
            
            # Random selection with bias toward smaller values
            if len(candidates) > 1:
                # Weighted random: smaller values more likely
                weights = [1.0 / (c - base + 1) for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                pick = candidates[-1]
                for i, c in enumerate(candidates):
                    cumulative += weights[i]
                    if r <= cumulative:
                        pick = c
                        break
            else:
                pick = candidates[0]
            
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
        
        if not success:
            break
        rows.append(row)
    
    if success and len(rows) == n_rows:
        return rows
    return None

def _backtrack_with_forward_checking(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = []
    
    def solve(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            if pos + 1 < k + 1:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                has_next = False
                for next_val in range(next_lo, next_hi + 1):
                    next_new_diffs = {next_val - a for a in current_row + [val]}
                    if len(next_new_diffs) == len(current_row) + 1:
                        if not (next_new_diffs & (used_diffs | new_diffs)):
                            has_next = True
                            break
                if not has_next:
                    continue
            
            candidates.append(val)
        
        if not candidates:
            return False
        
        def score(val):
            new_diffs = {val - a for a in current_row}
            s1 = val * 10
            s2 = 0
            for d in new_diffs:
                s2 += sum(1 for ud in used_diffs if abs(ud - d) <= 3)
            return s1 + s2
        
        candidates.sort(key=score)
        
        if rng.random() < 0.3 and len(candidates) > 1:
            swap_idx = rng.randrange(len(candidates))
            candidates[0], candidates[swap_idx] = candidates[swap_idx], candidates[0]
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if solve(0, 0, []):
        return rows
    return None

def _generate_candidate_rows(n_rows, k, max_scope, rng, num_candidates_per_row=20):
    """
    Generate candidate rows for each row position using randomized greedy.
    Returns a list of lists, where each inner list contains candidate rows for that position.
    """
    candidates = [[] for _ in range(n_rows)]
    
    for attempt in range(num_candidates_per_row * 3):
        if time.time() > deadline - 5:
            break
        
        # Try to build a complete solution with randomized greedy
        used = set()
        current_rows = []
        success = True
        
        for i in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                base = row[-1]
                # Randomized selection
                candidates_val = []
                search_hi = min(base + 50, max_scope + 1)
                for x in range(base + 1, search_hi + 1):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates_val.append(x)
                
                if not candidates_val:
                    success = False
                    break
                
                # Randomly pick from candidates with bias toward smaller
                if len(candidates_val) > 1:
                    weights = [1.0 / (c - base + 1) for c in candidates_val]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cumulative = 0
                    pick = candidates_val[-1]
                    for c, w in zip(candidates_val, weights):
                        cumulative += w
                        if r <= cumulative:
                            pick = c
                            break
                else:
                    pick = candidates_val[0]
                
                row.append(pick)
                used |= {pick - a for a in row[:-1]}
            
            if not success:
                break
            current_rows.append(row)
        
        if success and len(current_rows) == n_rows:
            # Add these rows to candidates
            for i in range(n_rows):
                if len(candidates[i]) < num_candidates_per_row:
                    # Check if this exact row isn't already in candidates
                    if current_rows[i] not in candidates[i]:
                        candidates[i].append(current_rows[i])
            
            # Early exit if we have enough candidates
            if all(len(c) >= num_candidates_per_row for c in candidates):
                break
    
    return candidates

def _hungarian_matching(candidates, n_rows, k, deadline):
    """
    Use Hungarian algorithm to find optimal assignment of candidate rows to positions.
    Cost is defined as the maximum scope of the assigned row, plus penalty for conflicts.
    """
    if n_rows == 0:
        return [], 0
    
    # Build cost matrix
    # rows: positions, cols: candidate indices
    # We need to handle variable number of candidates per position
    
    # First, flatten all candidates
    all_candidates = []
    pos_to_cand_idx = {}
    
    for pos in range(n_rows):
        pos_to_cand_idx[pos] = []
        for cand in candidates[pos]:
            all_candidates.append((pos, cand))
            pos_to_cand_idx[pos].append(len(all_candidates) - 1)
    
    n_cands = len(all_candidates)
    
    if n_cands == 0:
        return None, float('inf')
    
    # Build cost matrix: n_rows x n_cands
    # cost[i][j] = cost of assigning candidate j to position i
    # If candidate j is not valid for position i, cost = infinity
    
    INF = float('inf')
    cost_matrix = [[INF] * n_cands for _ in range(n_rows)]
    
    # Precompute diff sets for all candidates
    cand_diffs = []
    cand_maxes = []
    for pos, cand in all_candidates:
        dset = _row_diffs(cand)
        cand_diffs.append(dset)
        cand_maxes.append(max(cand))
    
    # Fill cost matrix
    for i in range(n_rows):
        for j in range(n_cands):
            pos, cand = all_candidates[j]
            if pos != i:
                continue
            
            # Base cost: max element
            cost = cand_maxes[j]
            
            # Check for conflicts with other candidates that might be assigned
            # This is tricky - we'll use a heuristic: count how many other candidates
            # share differences with this one
            conflict_count = 0
            for j2 in range(n_cands):
                if j2 == j:
                    continue
                pos2, _ = all_candidates[j2]
                if pos2 == i:
                    continue
                if cand_diffs[j] & cand_diffs[j2]:
                    conflict_count += 1
            
            # Add penalty for conflicts
            cost += conflict_count * 100
            
            cost_matrix[i][j] = cost
    
    # Solve Hungarian algorithm
    # scipy's linear_sum_assignment expects square matrix or can handle rectangular
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Check if assignment is valid
    assigned_rows = [None] * n_rows
    used_cands = set()
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i][j] >= INF:
            return None, float('inf')
        pos, cand = all_candidates[j]
        if pos != i:
            return None, float('inf')
        if j in used_cands:
            return None, float('inf')
        used_cands.add(j)
        assigned_rows[i] = cand
    
    if any(r is None for r in assigned_rows):
        return None, float('inf')
    
    # Validate the solution
    if not _validate(assigned_rows):
        return None, float('inf')
    
    score = _max_entry(assigned_rows)
    return assigned_rows, score

def _local_search_hungarian(best_rows, n_rows, k, rng, deadline):
    """
    Local search using Hungarian matching on neighborhoods of the current best solution.
    """
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(best_rows)
    
    # Generate neighborhood candidates by perturbing existing rows
    num_perturbations = 30
    candidates = [[] for _ in range(n_rows)]
    
    # Add original rows as candidates
    for i in range(n_rows):
        candidates[i].append(current_rows[i])
    
    # Generate perturbed versions
    for it in range(num_perturbations):
        if time.time() > deadline - 10:
            break
        
        # Pick a random row to perturb
        ri = rng.randrange(n_rows)
        row = current_rows[ri][:]
        
        # Pick a position to modify
        pi = rng.randrange(1, k + 1)
        old_val = row[pi]
        
        # Try nearby values
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else best_score + 10
        
        # Generate a few candidate values
        for _ in range(5):
            if lo > hi:
                break
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            new_row = row[:]
            new_row[pi] = new_val
            
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            
            # Check if this row is valid on its own (no internal duplicate diffs)
            dset = _row_diffs(new_row)
            expected = (len(new_row) * (len(new_row) - 1)) // 2
            if len(dset) != expected:
                continue
            
            # Add to candidates if not already present
            if new_row not in candidates[ri]:
                candidates[ri].append(new_row)
    
    # Use Hungarian matching to find best assignment
    rows, score = _hungarian_matching(candidates, n_rows, k, deadline)
    
    if rows is not None and score < best_score:
        return rows, score
    
    return current_rows, best_score

def _construct(n_rows, k):
    deadline = time.time() + 135  # Safety margin from 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get initial solution with greedy
    best_rows = _greedy_all(n_rows, k)
    if best_rows is None or not _validate(best_rows):
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
    
    best_score = _max_entry(best_rows)
    
    # Phase 2: Try randomized greedy with different seeds
    for attempt in range(10):
        if time.time() > deadline - 60:
            break
        
        rng_attempt = random.Random(42 + attempt)
        rows = _randomized_greedy(n_rows, k, rng_attempt, best_score, deadline)
        
        if rows is not None and _validate(rows):
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
    
    # Phase 3: Backtracking with forward checking (limited)
    if time.time() < deadline - 40:
        for bound in range(best_score - 1, max(best_score - 8, 100), -1):
            if time.time() > deadline - 40:
                break
            
            rows_bt = _backtrack_with_forward_checking(n_rows, k, bound, rng, deadline - 30)
            if rows_bt is not None and len(rows_bt) == n_rows:
                if _validate(rows_bt):
                    score = _max_entry(rows_bt)
                    if score < best_score:
                        best_score = score
                        best_rows = copy.deepcopy(rows_bt)
                    break
    
    # Phase 4: NEW - Hungarian matching based optimization
    if time.time() < deadline - 20:
        # Generate candidate rows
        max_scope = best_score + 10
        candidates = _generate_candidate_rows(n_rows, k, max_scope, rng, num_candidates_per_row=15)
        
        # Use Hungarian matching
        rows_h, score_h = _hungarian_matching(candidates, n_rows, k, deadline - 5)
        
        if rows_h is not None and score_h < best_score:
            best_score = score_h
            best_rows = copy.deepcopy(rows_h)
    
    # Phase 5: Local search with Hungarian refinement
    if time.time() < deadline - 5:
        rows_ls, score_ls = _local_search_hungarian(best_rows, n_rows, k, rng, deadline - 2)
        
        if score_ls < best_score:
            best_score = score_ls
            best_rows = copy.deepcopy(rows_ls)
    
    # Final validation
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k)
        if fresh_rows is not None and _validate(fresh_rows):
            best_rows = fresh_rows
        else:
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    for x in range(row[-1] + 1, 10000):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                best_rows.append(row)
    
    return best_rows

def construct(n_rows, k):
    return _construct(n_rows, k)
