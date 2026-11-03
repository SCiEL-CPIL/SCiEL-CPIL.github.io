import time
import random
import sys
import math
from collections import Counter

def _compute_diffs(row):
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, rng=None):
    if rng is None:
        rng = random
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
                used |= new_diffs
        rows.append(row)
    return rows, used

def _reconstruct_row_from_diffs(row_diffs, k):
    """Given a set of k*(k+1)/2 differences that form a valid difference set,
    reconstruct the row. The differences must include all pairwise differences
    of k+1 increasing integers starting at 0."""
    if len(row_diffs) != k * (k + 1) // 2:
        return None
    
    # The row has k+1 elements: 0 = a_0 < a_1 < ... < a_k
    # Differences are a_j - a_i for all i < j
    # We need to find the actual values.
    
    # Strategy: the differences include a_k - 0 = a_k (the largest),
    # and a_1 - 0 = a_1, a_k - a_{k-1}, etc.
    # 
    # Actually, let's use a different approach: we know the set of differences.
    # The row is determined by the differences in a specific way.
    # 
    # For k=6, we have 21 differences. The row has 7 elements.
    # 
    # Let's try: sort the differences. The smallest difference is a_1 (since a_0=0).
    # Actually no, the smallest difference could be a_j - a_{j-1} for any j.
    # 
    # Better approach: use the fact that the differences form a "difference basis."
    # 
    # Alternative: just do a greedy reconstruction.
    # We know a_0 = 0. The differences from a_0 are {a_1, a_2, ..., a_k}.
    # So a_1, ..., a_k are all in the difference set.
    # The remaining differences are a_j - a_i for i >= 1.
    #
    # Let's try: the k largest differences that are "compatible" with starting at 0
    # give us the row elements directly.
    
    # Actually, the simplest approach: the row elements are 0 and the k values
    # that, together with 0, produce exactly the given difference set.
    # 
    # For a difference set D of size k*(k+1)/2, the row is the unique (up to
    # translation, which is fixed by a_0=0) set of k+1 non-negative integers
    # whose pairwise differences are exactly D.
    #
    # This is a known problem. Let's use a greedy approach:
    # The smallest element is 0. The next element must be the smallest element
    # of D that can be part of a valid construction.
    
    D = sorted(row_diffs)
    if not D or D[0] <= 0:
        return None
    
    # Greedy: start with [0], repeatedly add the smallest unused difference
    # that is consistent with all existing elements.
    row = [0]
    used_diffs = set()
    
    for _ in range(k):
        found = False
        for d in D:
            if d in used_diffs:
                continue
            # Check if d can be added: d - a must be in D for all a in row
            # (or d must be the new max)
            candidate = d
            valid = True
            for a in row:
                diff = candidate - a
                if diff < 0:
                    valid = False
                    break
                if diff != 0 and diff not in row_diffs:
                    valid = False
                    break
            if valid and candidate > row[-1]:
                row.append(candidate)
                used_diffs.add(d)
                found = True
                break
        if not found:
            # Try larger candidates
            for d in D:
                if d in used_diffs:
                    continue
                if d <= row[-1]:
                    continue
                candidate = d
                valid = True
                for a in row:
                    diff = candidate - a
                    if diff < 0:
                        valid = False
                    elif diff != 0 and diff not in row_diffs:
                        valid = False
                        break
                if valid:
                    row.append(candidate)
                    used_diffs.add(d)
                    found = True
                    break
        if not found:
            return None
    
    if len(row) != k + 1:
        return None
    
    # Verify
    actual_diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            actual_diffs.add(row[j] - row[i])
    if actual_diffs != set(row_diffs):
        return None
    
    return row

def _row_diffs_to_list(row):
    """Return sorted list of all pairwise differences for a row."""
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return sorted(diffs)

def _slot_sa_optimize(rows, n_rows, k, deadline, rng=None):
    """
    Slot-based simulated annealing:
    - Representation: for each row, we store the multiset of differences (as a sorted list)
    - A move: pick a row, pick one of its differences, change it to a new value
    - Feasibility: all differences across all rows must be distinct
    - Objective: minimize the maximum element in any row (after reconstruction)
    """
    if rng is None:
        rng = random
    
    # Initialize: extract differences from each row
    row_diffs = []
    for row in rows:
        rd = _row_diffs_to_list(row)
        row_diffs.append(rd)
    
    # Build global difference set
    all_diffs = set()
    for rd in row_diffs:
        for d in rd:
            if d in all_diffs:
                # Shouldn't happen if rows are valid
                pass
            all_diffs.add(d)
    
    # Track scope
    current_scope = _get_scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # SA parameters
    T = 20.0
    cooling = 0.9997
    min_T = 0.01
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while T > min_T and time.time() < deadline:
        step += 1
        
        if step % 1000 == 0 and no_improve_steps > max_no_improve:
            # Restart with greedy
            new_rows, _ = _greedy_init(n_rows, k, rng=rng)
            if _is_valid(new_rows, k):
                new_scope = _get_scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    row_diffs = []
                    for row in new_rows:
                        row_diffs.append(_row_diffs_to_list(row))
                    current_scope = new_scope
                    T = 20.0
                    no_improve_steps = 0
            no_improve_steps = 0
            max_no_improve = 3000
        
        # Pick a row
        r_idx = rng.randint(0, n_rows - 1)
        rd = row_diffs[r_idx]
        
        # Pick a difference to modify
        d_idx = rng.randint(0, len(rd) - 1)
        old_d = rd[d_idx]
        
        # Generate new difference
        if rng.random() < 0.7:
            # Small perturbation
            new_d = old_d + rng.choice([-2, -1, 1, 2, 3, 4, 5])
        else:
            # Larger jump
            new_d = rng.randint(1, current_scope + 10)
        
        if new_d <= 0 or new_d == old_d:
            continue
        
        # Check if new_d conflicts with any existing difference
        if new_d in all_diffs:
            # Conflict: we'd need to also move the conflicting difference
            # For simplicity, skip this move (or do a swap)
            # Let's try a swap: find which row has new_d and swap
            swap_row = -1
            swap_idx = -1
            for ri in range(n_rows):
                for di in range(len(row_diffs[ri])):
                    if row_diffs[ri][di] == new_d:
                        swap_row = ri
                        swap_idx = di
                        break
                if swap_row >= 0:
                    break
            
            if swap_row >= 0:
                # Swap: row r_idx gets new_d, row swap_row gets old_d
                # But we need to check that old_d doesn't conflict in swap_row
                if old_d not in row_diffs[swap_row]:
                    row_diffs[r_idx][d_idx] = new_d
                    row_diffs[swap_row][swap_idx] = old_d
                    all_diffs.discard(old_d)
                    all_diffs.discard(new_d)
                    all_diffs.add(old_d)
                    all_diffs.add(new_d)
                    
                    # Try to reconstruct rows
                    new_rows = []
                    valid = True
                    for ri in range(n_rows):
                        nr = _reconstruct_row_from_diffs(row_diffs[ri], k)
                        if nr is None:
                            valid = False
                            break
                        new_rows.append(nr)
                    
                    if valid:
                        new_scope = _get_scope(new_rows)
                        # Cost: scope + penalty for any issues
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in new_rows]
                            current_scope = new_scope
                            rows = new_rows
                            no_improve_steps = 0
                        elif new_scope <= current_scope:
                            rows = new_rows
                            current_scope = new_scope
                            no_improve_steps = 0
                        else:
                            # Accept with probability
                            delta = new_scope - current_scope
                            if rng.random() < math.exp(-delta / max(T, min_T)):
                                rows = new_rows
                                current_scope = new_scope
                            else:
                                # Revert
                                row_diffs[r_idx][d_idx] = old_d
                                row_diffs[swap_row][swap_idx] = new_d
                                all_diffs.discard(old_d)
                                all_diffs.discard(new_d)
                                all_diffs.add(old_d)
                                all_diffs.add(new_d)
                                no_improve_steps += 1
                    else:
                        # Revert
                        row_diffs[r_idx][d_idx] = old_d
                        row_diffs[swap_row][swap_idx] = new_d
                        all_diffs.discard(old_d)
                        all_diffs.discard(new_d)
                        all_diffs.add(old_d)
                        all_diffs.add(new_d)
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        else:
            # No conflict: just change the difference
            row_diffs[r_idx][d_idx] = new_d
            all_diffs.discard(old_d)
            all_diffs.add(new_d)
            
            # Try to reconstruct the row
            nr = _reconstruct_row_from_diffs(row_diffs[r_idx], k)
            if nr is not None:
                new_rows = [list(r) for r in rows]
                new_rows[r_idx] = nr
                new_scope = _get_scope(new_rows)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    current_scope = new_scope
                    rows = new_rows
                    no_improve_steps = 0
                elif new_scope <= current_scope:
                    rows = new_rows
                    current_scope = new_scope
                    no_improve_steps = 0
                else:
                    delta = new_scope - current_scope
                    if rng.random() < math.exp(-delta / max(T, min_T)):
                        rows = new_rows
                        current_scope = new_scope
                    else:
                        # Revert
                        row_diffs[r_idx][d_idx] = old_d
                        all_diffs.discard(new_d)
                        all_diffs.add(old_d)
                        no_improve_steps += 1
            else:
                # Revert
                row_diffs[r_idx][d_idx] = old_d
                all_diffs.discard(new_d)
                all_diffs.add(old_d)
                no_improve_steps += 1
        
        T *= cooling
    
    return best_rows

def _direct_sa_optimize(rows, n_rows, k, deadline, rng=None):
    """
    Direct SA on row elements with a focus on scope compression.
    Different from the current program: uses a "push down" strategy where
    we specifically try to reduce the maximum element by adjusting the row
    that contains it, and uses a different neighborhood structure.
    """
    if rng is None:
        rng = random
    
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    T = 15.0
    cooling = 0.9998
    min_T = 0.05
    
    no_improve = 0
    max_no_improve = 3000
    
    while T > min_T and time.time() < deadline:
        if no_improve > max_no_improve:
            # Try a fresh greedy
            new_rows, _ = _greedy_init(n_rows, k, rng=rng)
            if _is_valid(new_rows, k):
                ns = _get_scope(new_rows)
                if ns < best_scope:
                    best_scope = ns
                    best_rows = [list(r) for r in new_rows]
                    rows = [list(r) for r in new_rows]
                    used = set()
                    for row in rows:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                used.add(row[j] - row[i])
                    T = 15.0
                    no_improve = 0
            no_improve = 0
            max_no_improve = 2000
        
        # Find the row with the maximum element
        max_scope = max(max(row) for row in rows)
        max_row_idx = -1
        for ri in range(n_rows):
            if max(rows[ri]) == max_scope:
                max_row_idx = ri
                break
        
        if rng.random() < 0.5 and max_row_idx >= 0:
            # Target the max row: try to reduce its last element
            row = rows[max_row_idx]
            old_last = row[-1]
            # Try to find a new last element that's smaller
            # Remove last element's differences
            for i in range(len(row) - 1):
                used.discard(row[-1] - row[i])
            
            # Try candidates from old_last-1 down to row[-2]+1
            found_new = False
            for new_last in range(old_last - 1, row[-2], -1):
                # Check if new_last - row[i] is available for all i < len(row)-1
                ok = True
                for i in range(len(row) - 1):
                    d = new_last - row[i]
                    if d in used:
                        ok = False
                        break
                if ok:
                    row[-1] = new_last
                    for i in range(len(row) - 1):
                        used.add(new_last - row[i])
                    found_new = True
                    break
            
            if found_new:
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                for i in range(len(row) - 1):
                    used.add(old_last - row[i])
                row[-1] = old_last
                no_improve += 1
        else:
            # Random element move
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Remove differences involving this element
            removed = []
            for i in range(len(row)):
                if i == pos:
                    continue
                if i < pos:
                    d = row[pos] - row[i]
                    used.discard(d)
                    removed.append((i, d))
                else:
                    d = row[i] - row[pos]
                    used.discard(d)
                    removed.append((i, d))
            
            # Generate new value
            if rng.random() < 0.5:
                new_val = rng.randint(1, max(1, old_val - 1))
            else:
                new_val = rng.randint(1, max_scope + 5)
            
            if new_val == old_val:
                # Revert
                for i, d in removed:
                    used.add(d)
                continue
            
            if pos > 0 and new_val <= row[pos - 1]:
                for i, d in removed:
                    used.add(d)
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                for i, d in removed:
                    used.add(d)
                continue
            
            # Check conflicts
            ok = True
            for i in range(len(row)):
                if i == pos:
                    continue
                if i < pos:
                    d = new_val - row[i]
                    if d in used:
                        ok = False
                        break
                else:
                    d = row[i] - new_val
                    if d in used:
                        ok = False
                        break
            
            if ok:
                row[pos] = new_val
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        used.add(new_val - row[i])
                    else:
                        used.add(row[i] - new_val)
                
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    no_improve = 0
                elif rng.random() < math.exp(-(new_scope - max_scope) / max(T, min_T)):
                    no_improve = 0
                else:
                    # Revert
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            used.discard(new_val - row[i])
                        else:
                            used.discard(row[i] - new_val)
                    row[pos] = old_val
                    for i, d in removed:
                        used.add(d)
                    no_improve += 1
            else:
                # Revert
                for i, d in removed:
                    used.add(d)
                no_improve += 1
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy inits with different seeds
    for seed in [42, 123, 456, 789, 101]:
        r = random.Random(seed)
        rows, _ = _greedy_init(n_rows, k, rng=r)
        if _is_valid(rows, k):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r2) for r2 in rows]
    
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, rng=rng)
        return rows
    
    # Phase 2: Slot-based SA (new approach)
    if time.time() < deadline - 30:
        slot_rows = _slot_sa_optimize(best_rows, n_rows, k, deadline - 30, rng)
        if _is_valid(slot_rows, k):
            slot_scope = _get_scope(slot_rows)
            if slot_scope < best_scope:
                best_scope = slot_scope
                best_rows = [list(r) for r in slot_rows]
    
    # Phase 3: Direct SA with push-down strategy (new approach)
    if time.time() < deadline - 5:
        direct_rows = _direct_sa_optimize(best_rows, n_rows, k, deadline - 5, rng)
        if _is_valid(direct_rows, k):
            direct_scope = _get_scope(direct_rows)
            if direct_scope < best_scope:
                best_scope = direct_scope
                best_rows = [list(r) for r in direct_rows]
    
    if not _is_valid(best_rows, k):
        # Fallback
        rows, _ = _greedy_init(n_rows, k, rng=rng)
        return rows
    
    return best_rows
