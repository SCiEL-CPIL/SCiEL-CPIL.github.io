import time
import random
import sys
from collections import defaultdict

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 138.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def compute_diffs(rows):
        """Compute all differences for each row."""
        row_diffs = []
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
            row_diffs.append(diffs)
        return row_diffs

    def validate_from_diffs(row_diffs):
        """Check if all differences across rows are distinct. Return True if valid."""
        seen = set()
        for diffs in row_diffs:
            for d in diffs:
                if d in seen:
                    return False
                seen.add(d)
        return True

    def targeted_descent(rows, row_diffs, max_iters=50000):
        """Local search to reduce scope by moving individual elements."""
        best_scope = get_scope(rows)
        rng = random.Random(42)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            for ri in range(n_rows):
                for pos in range(1, len(rows[ri]) - 1):
                    old_val = rows[ri][pos]
                    lo = rows[ri][pos - 1] + 1
                    hi = rows[ri][pos + 1] - 1 if pos < len(rows[ri]) - 1 else best_scope
                    
                    if hi <= lo:
                        continue
                    
                    # Try to find a better value
                    for new_val in range(lo, hi + 1):
                        if new_val == old_val:
                            continue
                        
                        # Check if this move is valid
                        rows[ri][pos] = new_val
                        new_row_diffs = set()
                        for i in range(len(rows[ri])):
                            for j in range(i + 1, len(rows[ri])):
                                new_row_diffs.add(rows[ri][j] - rows[ri][i])
                        
                        # Check for conflicts with other rows
                        conflict = False
                        for other_ri in range(n_rows):
                            if other_ri == ri:
                                continue
                            if new_row_diffs & row_diffs[other_ri]:
                                conflict = True
                                break
                        
                        if not conflict:
                            # Compute new scope
                            new_scope = max(max(r) for r in rows)
                            if new_scope < best_scope:
                                # Accept the move
                                row_diffs[ri] = new_row_diffs
                                best_scope = new_scope
                                improved = True
                                break
                            else:
                                rows[ri][pos] = old_val
                        else:
                            rows[ri][pos] = old_val
                
                if improved:
                    break
            
            if not improved:
                # No improvement found, try random perturbation
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(rows[ri]) - 1)
                old_val = rows[ri][pos]
                lo = rows[ri][pos - 1] + 1
                hi = rows[ri][pos + 1] - 1 if pos < len(rows[ri]) - 1 else best_scope + 10
                if hi > lo:
                    new_val = rng.randint(lo, hi)
                    if new_val != old_val:
                        rows[ri][pos] = new_val
                        new_row_diffs = set()
                        for i in range(len(rows[ri])):
                            for j in range(i + 1, len(rows[ri])):
                                new_row_diffs.add(rows[ri][j] - rows[ri][i])
                        
                        # Check validity
                        all_diffs = set()
                        valid = True
                        for other_ri in range(n_rows):
                            if other_ri == ri:
                                for d in new_row_diffs:
                                    if d in all_diffs:
                                        valid = False
                                        break
                                    all_diffs.add(d)
                            else:
                                for d in row_diffs[other_ri]:
                                    if d in new_row_diffs:
                                        valid = False
                                        break
                                if not valid:
                                    break
                                all_diffs |= row_diffs[other_ri]
                        
                        if valid:
                            row_diffs[ri] = new_row_diffs
                            new_scope = max(max(r) for r in rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                        else:
                            rows[ri][pos] = old_val
        
        return rows, row_diffs, best_scope

    rows = greedy_init()
    row_diffs = compute_diffs(rows)
    best_scope = get_scope(rows)
    best_rows = [r[:] for r in rows]
    best_row_diffs = [set(d) for d in row_diffs]

    rng = random.Random(42)

    # Simulated annealing with restarts
    max_restarts = 300
    steps_per_restart = 80000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbed
        if restart == 0:
            cur = [r[:] for r in best_rows]
            cur_diffs = [set(d) for d in best_row_diffs]
        else:
            cur = [r[:] for r in best_rows]
            cur_diffs = [set(d) for d in best_row_diffs]
            # Perturb
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)
                    # Recompute diffs for this row
                    cur_diffs[ri] = set()
                    for i in range(len(cur[ri])):
                        for j in range(i + 1, len(cur[ri])):
                            cur_diffs[ri].add(cur[ri][j] - cur[ri][i])

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]
            best_row_diffs = [set(d) for d in cur_diffs]

        # Annealing
        T = max(0.05, 1.0 - restart / max_restarts)
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 20
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope = full_validate_scope(cur)

            if new_scope is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                    best_row_diffs = [set(d) for d in cur_diffs]
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                    # Update cur_diffs
                    cur_diffs[ri] = set()
                    for i in range(len(cur[ri])):
                        for j in range(i + 1, len(cur[ri])):
                            cur_diffs[ri].add(cur[ri][j] - cur[ri][i])
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val

            T *= 0.99995

        # Run targeted descent on the current solution
        if time.time() < deadline - 10:
            cur, cur_diffs, cur_scope = targeted_descent(cur, cur_diffs, max_iters=20000)
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [r[:] for r in cur]
                best_row_diffs = [set(d) for d in cur_diffs]

    return best_rows
