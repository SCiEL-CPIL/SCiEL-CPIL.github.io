import time
import random
import math
from itertools import combinations

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        prev = 0
        for a in r[1:]:
            if a <= prev:
                return False
            prev = a
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _construct_row_backtrack(used_diffs, k, max_scope, rng, time_limit=0.5):
    """
    Backtrack to construct a row of k+1 elements starting at 0
    such that all pairwise differences are not in used_diffs
    and the row stays within max_scope.
    Returns row or None.
    """
    start = time.time()
    
    def backtrack(row, local_diffs, pos):
        if time.time() - start > time_limit:
            return None
        if pos == k + 1:
            return list(row)
        
        min_val = row[-1] + 1
        max_val = min(max_scope, row[-1] + max_scope)
        
        # Try candidates: prefer smaller values but allow randomness
        # Generate candidate range
        candidates = []
        for v in range(min_val, max_val + 1):
            # Check all new differences
            valid = True
            new_diffs = []
            for a in row:
                d = v - a
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                candidates.append((v, new_diffs))
        
        if not candidates:
            return None
        
        # Order: mostly small-first with some randomization
        if rng.random() < 0.7:
            # Try smallest first
            candidates.sort(key=lambda x: x[0])
        else:
            # Random order
            rng.shuffle(candidates)
        
        # Limit candidates to avoid explosion
        max_candidates = 20
        if len(candidates) > max_candidates:
            # Keep a mix: some small, some random
            small = candidates[:5]
            rest = candidates[5:]
            rng.shuffle(rest)
            candidates = small + rest[:max_candidates - 5]
        
        for v, new_diffs in candidates:
            row.append(v)
            local_diffs.update(new_diffs)
            result = backtrack(row, local_diffs, pos + 1)
            if result is not None:
                return result
            row.pop()
            local_diffs.difference_update(new_diffs)
        
        return None
    
    return backtrack([0], set(), 1)

def _sa_optimize(rows, k, deadline, rng):
    """
    Simulated Annealing to minimize scope while maintaining validity.
    Works on the rows directly, moving individual elements.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Track used differences for fast collision detection
    def build_used_diffs(rows):
        used = set()
        for r in rows:
            used.update(_row_diffs(r))
        return used
    
    used_diffs = build_used_diffs(rows)
    
    T = 10.0
    cooling = 0.99995
    min_T = 0.05
    
    while time.time() < deadline:
        # Pick random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)  # Don't move position 0 (always 0)
        
        row = rows[i]
        old_val = row[j]
        prev_val = row[j-1] if j > 0 else -1
        next_val = row[j+1] if j < k else float('inf')
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        # Propose new value
        if rng.random() < 0.6:
            # Try to move down
            if old_val > lo:
                step = rng.randint(1, max(1, (old_val - lo) // 3))
                new_val = old_val - step
            else:
                new_val = old_val
        else:
            # Random perturbation
            span = hi - lo
            if span <= 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.25)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        # Check if move is valid (no new collisions)
        # Remove old differences involving position j
        old_diffs = set()
        for t in range(k + 1):
            if t != j:
                old_diffs.add(abs(row[t] - old_val))
        
        # Add new differences
        new_diffs = set()
        for t in range(k + 1):
            if t != j:
                new_diffs.add(abs(row[t] - new_val))
        
        # Check for collisions
        # New diffs that were not in old_diffs must not be in used_diffs
        # Old diffs that are not in new_diffs must not be the only occurrence
        # This is complex, so let's just do a full validity check on the affected row
        # and check against global used_diffs
        
        # Quick check: new_diffs - old_diffs must not intersect used_diffs
        new_only = new_diffs - old_diffs
        if new_only & used_diffs:
            continue  # Collision, skip
        
        # Apply move
        row[j] = new_val
        new_scope = _scope(rows)
        
        # Acceptance criterion
        delta_scope = new_scope - best_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / max(T, 0.01)):
            # Update used_diffs
            used_diffs.difference_update(old_diffs)
            used_diffs.update(new_diffs)
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        else:
            # Revert
            row[j] = old_val
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _greedy_random(n_rows, k, rng, aggressiveness=0.5):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_search = last + 60
            while x <= max_search:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                    if len(candidates) >= 8:
                        break
                x += 1
            
            if not candidates:
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            
            power = 1.0 + aggressiveness
            weights = [1.0 / (c[0] - last + 1) ** power for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions using backtracking
    init_deadline = start_time + 15.0
    
    candidates = []
    attempt = 0
    while time.time() < init_deadline and attempt < 50:
        attempt += 1
        aggr = rng.uniform(0.2, 1.0)
        rows = _greedy_random(n_rows, k, rng, aggressiveness=aggr)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        
        # Also try backtracking-based construction
        if attempt % 3 == 0:
            used = set()
            bt_rows = []
            success = True
            for _ in range(n_rows):
                max_sc = int(best_scope * 0.95) if best_scope < float('inf') else 150
                row = _construct_row_backtrack(used, k, max_sc, rng, time_limit=0.3)
                if row is None:
                    success = False
                    break
                bt_rows.append(row)
                used.update(_row_diffs(row))
            
            if success and _valid(bt_rows, k):
                s = _scope(bt_rows)
                candidates.append((s, bt_rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in bt_rows]
    
    if not candidates:
        # Fallback
        rows = _greedy_random(n_rows, k, rng, aggressiveness=0.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            return [[0, 1, 2, 3, 4, 5]] * n_rows
    
    # Phase 2: Optimize top candidates with SA
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:4]
    
    remaining_time = deadline - time.time()
    if remaining_time < 10:
        return best_rows
    
    time_per_sa = remaining_time / len(top_candidates)
    
    for i, (s, rows) in enumerate(top_candidates):
        if time.time() >= deadline - 8:
            break
        
        sa_deadline = time.time() + time_per_sa * 0.92
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            actual_scope = _scope(sa_rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = sa_rows
    
    return best_rows
