import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # safety margin before 150s kill

    d = k * (k + 1) // 2  # differences per row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
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

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    # State variables for incremental search
    cur_rows = [r[:] for r in best_rows]
    
    # Initialize diff counts
    diff_counts = {}
    for row in cur_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                diff_counts[diff] = diff_counts.get(diff, 0) + 1
    
    # invalid_count: number of diff values with count != 1
    invalid_count = sum(1 for cnt in diff_counts.values() if cnt != 1)
    cur_scope = best_scope

    def perform_move(ri, pos, old_val, new_val):
        nonlocal invalid_count
        row = cur_rows[ri]
        
        # Remove old differences
        for j in range(len(row)):
            if j == pos:
                continue
            diff = abs(old_val - row[j])
            cnt = diff_counts.get(diff, 0)
            
            if cnt == 1:
                # 1 -> 0: Valid (1) becomes Invalid (0)
                invalid_count += 1
            elif cnt == 2:
                # 2 -> 1: Invalid (2) becomes Valid (1)
                invalid_count -= 1
            elif cnt == 0:
                # 0 -> -1: Should not happen, but if it does, Invalid (0) -> Invalid (-1? no, 0 is min)
                # Actually if cnt is 0, we are removing a diff that doesn't exist. Bug in logic?
                # No, we only remove diffs that exist. So cnt >= 1.
                pass
            
            diff_counts[diff] = cnt - 1
            if diff_counts[diff] == 0:
                del diff_counts[diff]

        # Add new differences
        for j in range(len(row)):
            if j == pos:
                continue
            diff = abs(new_val - row[j])
            cnt = diff_counts.get(diff, 0)
            
            if cnt == 0:
                # 0 -> 1: Invalid (0) becomes Valid (1)
                invalid_count -= 1
            elif cnt == 1:
                # 1 -> 2: Valid (1) becomes Invalid (2)
                invalid_count += 1
            # If cnt >= 2, Invalid stays Invalid
            
            diff_counts[diff] = cnt + 1

        row[pos] = new_val

    # Simulated Annealing
    T = 1.0
    while time.time() < deadline:
        ri = rng.randrange(n_rows)
        pos = rng.randint(1, len(cur_rows[ri]) - 1)
        old_val = cur_rows[ri][pos]
        
        lo = cur_rows[ri][pos - 1] + 1
        hi = cur_rows[ri][pos + 1] - 1 if pos < len(cur_rows[ri]) - 1 else best_scope + 20
        
        if hi <= lo:
            continue
        
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue

        perform_move(ri, pos, old_val, new_val)
        
        if invalid_count == 0:
            new_scope = get_scope(cur_rows)
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in cur_rows]
            
            delta = new_scope - cur_scope
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                cur_scope = new_scope
                # Accept move
            else:
                # Reject move, revert
                perform_move(ri, pos, new_val, old_val)
        else:
            # Invalid, revert
            perform_move(ri, pos, new_val, old_val)
        
        T *= 0.99995

    return best_rows
