import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & used:
            return False
        used |= diffs
    return True

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _generate_sidon_rows(k, max_entry, limit=None):
    """
    Generate all valid Sidon rows (k+1 increasing integers starting at 0, max <= max_entry)
    where all internal differences are distinct.
    Returns list of (row, diffs_set) tuples.
    """
    rows = []
    
    def build(pos, current, diffs):
        if limit is not None and len(rows) >= limit:
            return
        if time.time() > _gen_deadline:
            return
        if pos == k + 1:
            rows.append((current[:], diffs.copy()))
            return
        
        start = 1 if pos == 0 else current[-1] + 1
        for x in range(start, max_entry + 1):
            if pos == 0:
                if x != 0:
                    continue
            new_diffs = diffs.copy()
            ok = True
            for a in current:
                d = x - a
                if d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                current.append(x)
                build(pos + 1, current, new_diffs)
                current.pop()
    
    build(0, [0], set())
    return rows

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _mis_branch_and_bound(candidate_rows, n_rows, deadline):
    """
    Maximum Independent Set search on conflict graph.
    candidate_rows: list of (row, diffs_set) tuples
    Finds n_rows mutually compatible rows minimizing the maximum entry.
    
    Returns (selected_rows, max_entry) or (None, inf).
    """
    n = len(candidate_rows)
    if n < n_rows:
        return None, float('inf')
    
    # Build conflict matrix using bitmasks
    # Two rows conflict if they share any difference
    # Use a set of differences for each row
    row_diffs = [c[1] for c in candidate_rows]
    row_maxes = [max(c[0]) for c in candidate_rows]
    
    # Sort candidates by their max entry (ascending) to find small solutions first
    order = sorted(range(n), key=lambda i: (row_maxes[i], i))
    sorted_candidates = [candidate_rows[i] for i in order]
    sorted_diffs = [row_diffs[i] for i in order]
    sorted_maxes = [row_maxes[i] for i in order]
    
    # Build adjacency: adj[i] is a set of indices j > i that conflict with i
    # For efficiency, use difference sets
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_diffs[i] & sorted_diffs[j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Branch and bound
    best_solution = None
    best_max = float('inf')
    
    def bnb(selected, remaining, current_max):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        # If we've selected enough rows
        if len(selected) == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_solution = selected[:]
            return
        
        # Pruning: if current_max >= best_max, no need to continue
        if current_max >= best_max:
            return
        
        # Need more rows than available
        if len(selected) + len(remaining) < n_rows:
            return
        
        # Lower bound on final max: at least the max of selected so far,
        # and at least the (n_rows - len(selected))-th smallest max among remaining
        # Quick bound: the max must be at least the max of the next needed rows' minimum possible
        needed = n_rows - len(selected)
        if needed > 0 and remaining:
            # The max will be at least the needed-th smallest max in remaining
            # But we also need them to be compatible, so this is a lower bound
            min_possible_max = sorted_maxes[remaining[0]] if remaining else 0
            if min_possible_max > current_max:
                new_current_max = min_possible_max
            else:
                new_current_max = current_max
            if new_current_max >= best_max:
                return
        
        # Choose next candidate to try
        # Heuristic: try candidates with smallest max first
        # Filter remaining to those that don't conflict with selected
        compatible = []
        for idx in remaining:
            # Check if idx conflicts with any selected
            conflict = False
            for sel_idx in selected:
                if idx in adj[sel_idx]:
                    conflict = True
                    break
            if not conflict:
                compatible.append(idx)
        
        if not compatible:
            return
        
        # Sort compatible by max entry
        compatible.sort(key=lambda i: sorted_maxes[i])
        
        # Try each compatible candidate
        for idx in compatible:
            if time.time() > deadline:
                return
            
            new_max = max(current_max, sorted_maxes[idx])
            if new_max >= best_max:
                break  # Since sorted, all subsequent will be >=
            
            new_selected = selected + [idx]
            # Remaining: those after idx in compatible list that don't conflict with idx
            new_remaining = []
            for j in compatible:
                if j == idx:
                    continue
                if j not in adj[idx]:
                    new_remaining.append(j)
            
            bnb(new_selected, new_remaining, new_max)
    
    bnb([], list(range(n)), 0)
    
    if best_solution is not None:
        result_rows = [sorted_candidates[i][0] for i in best_solution]
        return result_rows, best_max
    return None, float('inf')

def _local_search_refine(rows, k, deadline):
    """
    Local search to reduce the maximum entry further.
    Tries to decrease individual entries while maintaining validity.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    all_used = set()
    for dset in row_diffs:
        all_used |= dset
    
    best_val = current_val
    best_rows = [r[:] for r in current_rows]
    
    T = 50.0
    T_min = 0.1
    cool = 0.999995
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(40, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        # Generate candidates: mostly decreasing, some increasing for exploration
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        # Small increasing for exploration
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        if random.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Compute new diffs for this row
            new_diffs = set()
            valid = True
            for i in range(len(cand_row)):
                for j in range(i + 1, len(cand_row)):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check cross-row conflicts
            other_diffs = all_used - row_diffs[ri]
            if new_diffs & other_diffs:
                continue
            
            # Compute new global max
            cand_row_max = max(cand_row)
            new_global_max = max([cand_row_max] + [max(current_rows[j]) for j in range(n_rows) if j != ri])
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
    
    return best_rows, best_val

def construct(n_rows, k):
    global _gen_deadline
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Phase 1: Greedy initialization
    greedy_rows = _greedy_init(n_rows, k)
    greedy_val = _global_max(greedy_rows)
    
    best_rows = [r[:] for r in greedy_rows]
    best_val = greedy_val
    
    # Phase 2: MIS branch-and-bound with increasing max_entry bounds
    # Start from the counting bound (105 for 7,5) up to greedy_val
    lower_bound = 105 if (n_rows == 7 and k == 5) else max(1, int(math.sqrt(n_rows * (k + 1) * k / 2)))
    
    # Try to find a solution with max_entry <= best_val - 1, best_val - 2, ...
    # Use MIS B&B on candidate rows with bounded max
    
    mis_deadline = start_time + 60.0  # Spend up to 60s on MIS
    
    # Generate candidates in batches by max_entry
    for max_entry in range(min(greedy_val - 1, 130), max(lower_bound - 1, 50), -1):
        if time.time() > mis_deadline:
            break
        
        # Generate Sidon rows with max entry <= max_entry
        # Limit generation time
        _gen_deadline = time.time() + 5.0
        candidates = _generate_sidon_rows(k, max_entry, limit=5000)
        
        if len(candidates) < n_rows:
            continue
        
        # Run MIS B&B
        result_rows, result_max = _mis_branch_and_bound(candidates, n_rows, time.time() + 10.0)
        
        if result_rows is not None:
            if result_max < best_val:
                best_val = result_max
                best_rows = [r[:] for r in result_rows]
            break  # Found solution at this bound
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 5:
        refined_rows, refined_val = _local_search_refine(best_rows, k, deadline - 3.0)
        if refined_val < best_val:
            best_val = refined_val
            best_rows = [r[:] for r in refined_rows]
    
    # Final validation
    if not _is_valid(best_rows):
        if _is_valid(greedy_rows):
            return greedy_rows
        fallback_row = [0, 1, 3, 7, 15, 31]
        return [fallback_row[:] for _ in range(n_rows)]
    
    return best_rows

# EVOLVE-BLOCK-END
