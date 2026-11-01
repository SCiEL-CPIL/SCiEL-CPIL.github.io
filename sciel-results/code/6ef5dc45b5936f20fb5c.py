import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

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

    def build_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    used.add(row[j] - ri)
        return used

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

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    best_used = build_used_set(best_rows)

    rng = random.Random(42)

    def targeted_descent(rows, used, budget):
        """Try to reduce the scope by shrinking the largest mark in the row that has it.
        Returns (improved_rows, new_scope, new_used) or (None, None, None) if no improvement found within budget."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_used = set(used)
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            # Find the row(s) containing the current max
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            # The last element is the max of this row
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                continue

            # Try a set of candidate values: prefer values just above the lower bound
            # to maximize the reduction, but also sample a few random ones
            candidates = set()
            # Sequential near the bottom (best reduction)
            for v in range(lower, min(lower + 15, old_val)):
                candidates.add(v)
            # Some random samples in the middle
            for _ in range(10):
                candidates.add(rng.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):  # try larger reductions first
                if new_val >= old_val:
                    continue
                
                # Incremental check
                # Remove old diffs involving old_val
                removed = {old_val - a for a in row[:-1]}
                # Add new diffs involving new_val
                added = {new_val - a for a in row[:-1]}
                
                # Check if any added diff is in (cur_used - removed)
                conflict = added & (cur_used - removed)
                if not conflict:
                    # Valid move
                    cur_used -= removed
                    cur_used |= added
                    cur[r][pos] = new_val
                    new_scope = get_scope(cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        break
                    else:
                        # Revert
                        cur_used -= added
                        cur_used |= removed
                        cur[r][pos] = old_val

            if not improved:
                # Try reducing a non-last element in the max row
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(5):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    
                    # Incremental check for middle element
                    # Diffs removed: nv_old - a for a in row[:pos2] and a - nv_old for a in row[pos2+1:]
                    # Diffs added: nv_new - a for a in row[:pos2] and a - nv_new for a in row[pos2+1:]
                    removed = set()
                    added = set()
                    for i in range(pos2):
                        removed.add(old2 - row[i])
                        added.add(nv - row[i])
                    for j in range(pos2 + 1, len(row)):
                        removed.add(row[j] - old2)
                        added.add(row[j] - nv)
                    
                    conflict = added & (cur_used - removed)
                    if not conflict:
                        cur_used -= removed
                        cur_used |= added
                        cur[r][pos2] = nv
                        new_scope = get_scope(cur)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            improved = True
                            break
                        else:
                            cur_used -= added
                            cur_used |= removed
                            cur[r][pos2] = old2
                if not improved:
                    break

        if improved:
            return cur, cur_scope, cur_used
        return None, None, None

    # Simulated annealing with restarts
    max_restarts = 300
    steps_per_restart = 80000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbed
        if restart == 0:
            cur = [r[:] for r in best_rows]
            cur_used = set(best_used)
        else:
            cur = [r[:] for r in best_rows]
            cur_used = set(best_used)
            # Perturb: random row, random mark, random new value
            # Note: Perturbation might break validity, so we need to handle that.
            # Simplest: try to apply a valid move as perturbation? Or just accept invalid start and let SA fix it?
            # Given the speed, we can afford to check validity after perturbation.
            perturbed = False
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 10
                if hi > lo:
                    new_val = rng.randint(lo, hi)
                    if new_val == cur[ri][pos]:
                        continue
                    old_val = cur[ri][pos]
                    
                    removed = set()
                    added = set()
                    if pos == len(cur[ri]) - 1:
                        for i in range(pos):
                            removed.add(old_val - cur[ri][i])
                            added.add(new_val - cur[ri][i])
                    else:
                        for i in range(pos):
                            removed.add(old_val - cur[ri][i])
                            added.add(new_val - cur[ri][i])
                        for j in range(pos + 1, len(cur[ri])):
                            removed.add(cur[ri][j] - old_val)
                            added.add(cur[ri][j] - new_val)
                    
                    conflict = added & (cur_used - removed)
                    if not conflict:
                        cur_used -= removed
                        cur_used |= added
                        cur[ri][pos] = new_val
                        perturbed = True
                    # If conflict, skip this perturbation step

            if not perturbed:
                continue

        cur_scope = get_scope(cur)
        # We assume cur is valid because we only made valid moves from a valid state
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]
            best_used = set(cur_used)

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

            # Incremental validation
            removed = set()
            added = set()
            
            if pos == len(cur[ri]) - 1:
                # Last element
                for i in range(pos):
                    removed.add(old_val - cur[ri][i])
                    added.add(new_val - cur[ri][i])
            else:
                # Middle element
                for i in range(pos):
                    removed.add(old_val - cur[ri][i])
                    added.add(new_val - cur[ri][i])
                for j in range(pos + 1, len(cur[ri])):
                    removed.add(cur[ri][j] - old_val)
                    added.add(cur[ri][j] - new_val)
            
            conflict = added & (cur_used - removed)
            
            if not conflict:
                # Valid move
                cur_used -= removed
                cur_used |= added
                cur[ri][pos] = new_val
                
                new_scope = get_scope(cur)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                    best_used = set(cur_used)
                
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    # Revert
                    cur_used -= added
                    cur_used |= removed
                    cur[ri][pos] = old_val
            else:
                # Invalid: reject
                pass

            T *= 0.99995

        # After annealing, run a short targeted descent to try to squeeze scope further
        if time.time() < deadline - 5.0:
            desc_rows, desc_scope, desc_used = targeted_descent(cur, cur_used, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                best_used = desc_used

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
