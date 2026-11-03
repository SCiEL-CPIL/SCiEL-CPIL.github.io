import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
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

def _fast_greedy(k, n_rows, rng, bias_small=True):
    """
    Generates a valid set of rows using a greedy approach.
    bias_small: if True, tries to pick the smallest valid number.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            # Try a window of candidates
            # If bias_small, start from x+1 and go up.
            # If we want diversity, we can start from x+1 + random_offset
            start = x + 1
            if not bias_small and rng.random() < 0.3:
                start = x + 1 + rng.randint(1, 5)
            
            for t in range(start, 200): # Limit search range to avoid infinite loops in bad states
                # Check if t creates conflicts
                ok = True
                for a in row:
                    d = t - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(t)
                    for a in row[:-1]:
                        used.add(t - a)
                    x = t
                    found = True
                    break
            
            if not found:
                # Fallback: find any valid t
                t = start
                while True:
                    ok = True
                    for a in row:
                        d = t - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        row.append(t)
                        for a in row[:-1]:
                            used.add(t - a)
                        x = t
                        break
                    t += 1
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _sa_optimize(rows, k, rng, deadline, phase="mixed"):
    """
    Optimizes the set of rows using Simulated Annealing.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(dc):
        collisions = sum(max(0, c - 1) for c in dc.values())
        # We need current scope, but passing it is easier
        return collisions, current_scope

    # Initial cost
    collisions, scope = get_cost(diff_counts)
    if phase == "collision":
        cost = 1000 * collisions + scope
    elif phase == "scope":
        cost = scope + 100 * collisions # Heavy penalty for collisions
    else:
        cost = 100 * collisions + scope

    best_cost = cost
    best_collisions = collisions
    best_scope_val = scope

    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    it = 0
    while time.time() < deadline:
        it += 1
        r = rng.random()
        
        if r < 0.3:
            # Move 1: Rebuild a random row
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            # Rebuild row greedily with randomness
            new_row = [0]
            x = 0
            # We want to keep scope small, so bias towards smaller numbers
            # But allow some randomness to escape local minima
            for _ in range(k):
                found = False
                # Try a range of candidates
                # Start from x+1, go up to current_scope + 5
                max_search = current_scope + 10
                # If we are in 'scope' phase, be more aggressive about small numbers
                search_range = 50 if phase == "scope" else 200
                
                for t in range(x + 1, x + 1 + search_range):
                    if t > max_search:
                        break
                    ok = True
                    for a in new_row:
                        d = t - a
                        if diff_counts.get(d, 0) > 0:
                            ok = False
                            break
                    if ok:
                        # Accept with some probability or just accept if valid
                        # For greedy rebuild, we usually accept the first valid one
                        # But to add noise, we can pick randomly from a few valid ones
                        new_row.append(t)
                        for a in new_row[:-1]:
                            d = t - a
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        x = t
                        found = True
                        break
                
                if not found:
                    # Fallback: find any valid t
                    t = x + 1
                    while True:
                        ok = True
                        for a in new_row:
                            d = t - a
                            if diff_counts.get(d, 0) > 0:
                                ok = False
                                break
                        if ok:
                            new_row.append(t)
                            for a in new_row[:-1]:
                                d = t - a
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                            x = t
                            break
                        t += 1
            
            new_scope = _scope(current_rows) # Need to update scope logic? 
            # Actually, scope is global. Let's just calculate cost.
            
            new_collisions = sum(max(0, c - 1) for c in diff_counts.values())
            # Update global scope for cost calculation
            # This is expensive. Let's maintain a max_val tracker?
            # For now, calculating scope is O(n*k). k=6, n=7. Very fast.
            cur_scope = _scope(current_rows)
            
            if phase == "collision":
                new_cost = 1000 * new_collisions + cur_scope
            elif phase == "scope":
                new_cost = cur_scope + 100 * new_collisions
            else:
                new_cost = 100 * new_collisions + cur_scope

            if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / max(T, min_T)):
                current_rows[i] = new_row
                cost = new_cost
                current_scope = cur_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in current_rows]
                    best_scope = cur_scope
            else:
                # Reject: revert
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                # Cost and scope revert implicitly

        elif r < 0.7:
            # Move 2: Compress a random element
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k) # Don't move 0
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            
            # Remove row diffs
            _remove_row_diffs(diff_counts, current_rows[i])
            
            found_new = False
            # Try to find the smallest valid value > prev_val
            # This directly reduces scope if old_val is large
            # We search from prev_val + 1 up to old_val
            # If we find a smaller valid value, we accept it (likely)
            # If we don't, we might try a random value in [prev_val+1, old_val-1]
            
            best_candidate = old_val
            # We want to minimize the value, so we search upwards from prev_val+1
            # But we also want to allow moves to larger values if it helps overall?
            # Usually, compression is good.
            
            for new_val in range(prev_val + 1, old_val + 1):
                if new_val == old_val:
                    break
                # Check validity of new_val in this row
                ok = True
                for t in range(k + 1):
                    if t == j:
                        continue
                    d = abs(new_val - current_rows[i][t])
                    if d == 0:
                        ok = False
                        break
                    # Check if d is in diff_counts (other rows)
                    if diff_counts.get(d, 0) > 0:
                        ok = False
                        break
                    # Check if d is generated by other pairs in this row
                    # We removed the row, so diff_counts only has other rows.
                    # We must check internal conflicts.
                    # The internal conflicts are differences between new_val and other elements.
                    # We need to ensure these new differences don't collide with each other.
                    # Since we are only changing one element, the new diffs are {new_val - a_t for t != j}.
                    # These must be distinct.
                    # And they must not be in diff_counts.
                    pass # The simple check above is not sufficient for internal conflicts.
                
                # Let's do a proper check for internal conflicts
                # This is O(k) per candidate.
                # Since k=6, it's fine.
                
                # Re-do check properly
                new_diffs_in_row = set()
                conflict_internal = False
                conflict_external = False
                for t in range(k + 1):
                    if t == j:
                        continue
                    d = abs(new_val - current_rows[i][t])
                    if d in new_diffs_in_row:
                        conflict_internal = True
                        break
                    new_diffs_in_row.add(d)
                    if diff_counts.get(d, 0) > 0:
                        conflict_external = True
                        break
                
                if not conflict_internal and not conflict_external:
                    best_candidate = new_val
                    break # Found the smallest valid one
            
            if best_candidate != old_val:
                current_rows[i][j] = best_candidate
                _add_row_diffs(diff_counts, current_rows[i])
                
                new_collisions = sum(max(0, c - 1) for c in diff_counts.values())
                cur_scope = _scope(current_rows)
                
                if phase == "collision":
                    new_cost = 1000 * new_collisions + cur_scope
                elif phase == "scope":
                    new_cost = cur_scope + 100 * new_collisions
                else:
                    new_cost = 100 * new_collisions + cur_scope

                if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / max(T, min_T)):
                    cost = new_cost
                    current_scope = cur_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(r) for r in current_rows]
                        best_scope = cur_scope
                else:
                    # Revert
                    _remove_row_diffs(diff_counts, current_rows[i])
                    current_rows[i][j] = old_val
                    _add_row_diffs(diff_counts, current_rows[i])
            else:
                # No change found, revert row diffs
                _add_row_diffs(diff_counts, current_rows[i])

        else:
            # Move 3: Random Perturbation
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                _add_row_diffs(diff_counts, current_rows[i]) # No-op, just to be safe if we removed? No, we didn't remove yet.
                continue
                
            # Pick a random new value
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            
            # Check if this move is even valid internally? 
            # If not, we can still accept it with low probability in SA to escape, 
            # but the cost will be high.
            
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_collisions = sum(max(0, c - 1) for c in diff_counts.values())
            cur_scope = _scope(current_rows)
            
            if phase == "collision":
                new_cost = 1000 * new_collisions + cur_scope
            elif phase == "scope":
                new_cost = cur_scope + 100 * new_collisions
            else:
                new_cost = 100 * new_collisions + cur_scope

            if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / max(T, min_T)):
                cost = new_cost
                current_scope = cur_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in current_rows]
                    best_scope = cur_scope
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])

        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0 # Safety margin
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    candidates = []
    
    # Generate diverse initial candidates
    # 1. Standard greedy (small bias)
    candidates.append(_fast_greedy(k, n_rows, rng, bias_small=True))
    # 2. Greedy with some spread
    candidates.append(_fast_greedy(k, n_rows, rng, bias_small=False))
    
    # 3. Randomized greedy
    for _ in range(3):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                # Try a random window
                for t in range(x + 1, x + 50):
                    ok = True
                    for a in row:
                        d = t - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        row.append(t)
                        for a in row[:-1]:
                            used.add(t - a)
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    while True:
                        ok = True
                        for a in row:
                            d = t - a
                            if d in used:
                                ok = False
                                break
                        if ok:
                            row.append(t)
                            for a in row[:-1]:
                                used.add(t - a)
                            x = t
                            break
                        t += 1
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    # Phase 1: Collision Resolution
    # Run SA with high penalty for collisions
    for idx, cand in enumerate(candidates[:4]):
        if time.time() > _DEADLINE - 10:
            break
        sa_deadline = time.time() + 5
        s, rows = _sa_optimize(cand, k, rng, sa_deadline, phase="collision")
        consider(rows)
        
    # Phase 2: Scope Optimization
    if _BEST is not None and time.time() < _DEADLINE - 5:
        remaining_time = _DEADLINE - time.time()
        best_rows_to_optimize = _BEST[1]
        
        # Run SA focusing on scope
        # We can run multiple passes
        while time.time() < _DEADLINE - 2:
            sa_deadline = time.time() + min(10, _DEADLINE - time.time() - 2)
            s, rows = _sa_optimize(best_rows_to_optimize, k, rng, sa_deadline, phase="scope")
            consider(rows)
            if _BEST is not None:
                best_rows_to_optimize = _BEST[1]
            else:
                break

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
