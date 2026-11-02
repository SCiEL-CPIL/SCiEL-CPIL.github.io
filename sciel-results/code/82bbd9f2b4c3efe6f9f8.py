"""Simulated-annealing search for difference-triangle sets.

Strategy: Treat the entire collection of n_rows rows as a single state in a
combinatorial optimization landscape. Use simulated annealing with:
- Energy = scope (max element) + large penalty for duplicate differences
- Moves: pick a random row, pick a random position, try a new value
- Acceptance: Metropolis criterion with cooling schedule
- Restart: If stuck, reinitialize with greedy + randomization

This is structurally different from the current approach which builds rows
independently with greedy heuristics and then applies local compaction.
"""

import time
import random
import sys
from array import array


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 115.0
    m = k + 1  # entries per row

    def greedy_init(rng):
        """Initialize with greedy construction plus randomization."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                # Random skip for variety
                if rng.random() < 0.15:
                    continue
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    def compute_energy(rows):
        """Compute energy: scope + conflict penalty.
        Returns (scope, n_conflicts)."""
        all_diffs = {}
        n_conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        n_conflicts += 1
                    else:
                        all_diffs[d] = 1
        scope = max(max(r) for r in rows) if rows else 0
        return scope, n_conflicts

    def try_move(rows, rng, scope_limit):
        """Try a single random move. Returns (new_rows, delta_scope, delta_conflicts) or None."""
        r_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, m)  # Don't move the first element (must be 0)
        row = rows[r_idx]
        old_val = row[pos]
        
        # Determine range for new value
        min_val = row[pos - 1] + 1
        max_val = min(scope_limit + 5, old_val + 10)
        
        if min_val >= max_val:
            return None
        
        new_val = rng.randint(min_val, max_val)
        if new_val == old_val:
            return None
        
        # Compute difference changes
        old_diffs = set()
        new_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, m):
            old_diffs.add(row[j] - old_val)
            new_diffs.add(row[j] - new_val)
        
        # Check for internal conflicts in new_diffs
        if len(new_diffs) != len(new_diffs):
            return None
        
        # Estimate delta conflicts: check if new_diffs elements are in global used
        # (approximation: we don't track global used here, so we rely on energy recompute)
        
        rows[r_idx][pos] = new_val
        return rows

    def local_compact(rows):
        """Right-to-left compaction pass."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])
        
        improved = True
        max_iters = 10
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    if current_val <= row[pos - 1] + 1:
                        continue
                    new_val = current_val - 1
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(current_val - row[i])
                    for j in range(pos + 1, len(row)):
                        old_diffs.add(row[j] - current_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    
                    conflict = False
                    for d in new_diffs:
                        if d in old_diffs:
                            continue
                        if d in all_diffs:
                            conflict = True
                            break
                    if not conflict:
                        for d in old_diffs:
                            all_diffs.discard(d)
                        for d in new_diffs:
                            all_diffs.add(d)
                        row[pos] = new_val
                        improved = True
        return rows

    # Initial solution: greedy
    rng = random.Random(42)
    rows = greedy_init(rng)
    rows = local_compact(rows)
    best_scope, best_conflicts = compute_energy(rows)
    best_rows = [list(r) for r in rows]
    
    if best_conflicts == 0:
        current_best_scope = best_scope
    else:
        current_best_scope = best_scope + best_conflicts * 1000  # Penalize conflicts

    # Simulated annealing
    T = 50.0
    T_min = 0.5
    alpha = 0.9995  # Cooling rate
    steps_per_restart = 5000
    
    while time.time() < deadline:
        rows = [list(r) for r in best_rows]
        current_scope, current_conflicts = compute_energy(rows)
        current_energy = current_scope + current_conflicts * 1000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break
            
            # Try a move
            move_result = try_move(rows, rng, current_scope + 10)
            if move_result is None:
                continue
            
            new_rows = move_result
            new_scope, new_conflicts = compute_energy(new_rows)
            new_energy = new_scope + new_conflicts * 1000
            
            delta = new_energy - current_energy
            
            # Metropolis acceptance
            if delta < 0 or (T > T_min and rng.random() < min(1.0, exp(-delta / T))):
                rows = new_rows
                current_energy = new_energy
                current_scope, current_conflicts = new_scope, new_conflicts
                
                # Update best if valid and better
                if current_conflicts == 0 and current_scope < current_best_scope:
                    current_best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            
            # Cool down
            T = max(T_min, T * alpha)
        
        # Periodic local compaction on best
        if step % 1000 == 0:
            best_rows = local_compact(best_rows)
            bs, bc = compute_energy(best_rows)
            if bc == 0 and bs < current_best_scope:
                current_best_scope = bs

    # Final compaction
    best_rows = local_compact(best_rows)
    
    # Verify and return
    used = set()
    valid = True
    for row in best_rows:
        if len(row) != m or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    
    if not valid:
        # Fallback to deterministic greedy
        u = set()
        out = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < m:
                x += 1
                nd = {x - a for a in row}
                if not (nd & u):
                    row.append(x)
                    u |= nd
            out.append(row)
        return out
    
    return best_rows


def exp(x):
    """Fast exponential for Metropolis criterion."""
    if x < -100:
        return 0.0
    if x > 100:
        return 1e43
    # Use math.exp but avoid import overhead in tight loop by approximating
    # For small |x|, use Taylor; for larger, use built-in
    import math
    return math.exp(x)


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
