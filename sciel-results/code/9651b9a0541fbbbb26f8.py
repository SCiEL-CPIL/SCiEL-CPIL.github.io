"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def greedy_construct(n_rows, k, rng, noise=0):
        """
        Construct rows greedily. If noise > 0, sometimes pick a larger value
        to escape local minima in the greedy choice.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # If noise is present, we might skip some valid small values
                # to create more diverse initial solutions.
                # But strictly, we need to find *a* valid value.
                # Let's stick to finding the smallest valid value to ensure validity,
                # but we can randomize the *order* of rows or add small perturbations later.
                # For now, standard greedy is a good baseline.
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def local_search_sum(rows, rng, max_iters=5000):
        """
        Local search to minimize the sum of all elements.
        Moves: 
        1. Shrink an element (if possible without conflict).
        2. Shift an element (if possible without conflict).
        """
        rows = [list(r) for r in rows]
        n = len(rows)
        cur_sum = sum(sum(r) for r in rows)
        best_sum = cur_sum
        best_rows = [list(r) for r in rows]
        
        # Pre-compute used diffs for O(1) lookup
        max_val = max(max(r) for r in rows)
        used = bytearray(max_val + 2)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used[d] = 1

        T = 1.0
        for it in range(max_iters):
            if _time_left() < 5.0:
                break
                
            move_type = rng.random()
            if move_type < 0.5:
                # Try to shrink a non-first element
                r_idx = rng.randrange(n)
                row = rows[r_idx]
                if len(row) <= 1: continue
                p_idx = rng.randint(1, len(row) - 1)
                old_val = row[p_idx]
                # Try to reduce to old_val - 1, or random smaller
                # We must maintain increasing order: row[p_idx-1] < new_val < row[p_idx+1] (if exists)
                lo = row[p_idx - 1] + 1
                hi = row[p_idx + 1] - 1 if p_idx < len(row) - 1 else old_val - 1
                if hi < lo:
                    continue
                # Pick a new value in [lo, old_val-1]
                if old_val <= lo:
                    continue
                new_val = rng.randint(lo, old_val - 1)
                
                # Check conflicts
                old_diffs = []
                new_diffs = []
                for j in range(len(row)):
                    if j == p_idx: continue
                    if j < p_idx:
                        old_diffs.append(old_val - row[j])
                        new_diffs.append(new_val - row[j])
                    else:
                        old_diffs.append(row[j] - old_val)
                        new_diffs.append(row[j] - new_val)
                
                # Remove old
                for d in old_diffs:
                    used[d] = 0
                # Check new
                conflict = False
                for d in new_diffs:
                    if d >= len(used):
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        conflict = True
                        break
                if conflict:
                    for d in old_diffs:
                        used[d] = 1
                    continue
                
                # Apply
                for d in new_diffs:
                    used[d] = 1
                row[p_idx] = new_val
                cur_sum += new_val - old_val
                
                if cur_sum < best_sum:
                    best_sum = cur_sum
                    best_rows = [list(r) for r in rows]
            else:
                # Try to shift an element (increase or decrease)
                r_idx = rng.randrange(n)
                row = rows[r_idx]
                if len(row) <= 1: continue
                p_idx = rng.randint(1, len(row) - 1)
                old_val = row[p_idx]
                
                # Random step
                step = rng.randint(-5, 5)
                new_val = old_val + step
                lo = row[p_idx - 1] + 1
                hi = row[p_idx + 1] - 1 if p_idx < len(row) - 1 else max_val + 10
                if new_val < lo or new_val > hi or new_val == old_val:
                    continue
                
                old_diffs = []
                new_diffs = []
                for j in range(len(row)):
                    if j == p_idx: continue
                    if j < p_idx:
                        old_diffs.append(old_val - row[j])
                        new_diffs.append(new_val - row[j])
                    else:
                        old_diffs.append(row[j] - old_val)
                        new_diffs.append(row[j] - new_val)
                
                for d in old_diffs:
                    used[d] = 0
                conflict = False
                for d in new_diffs:
                    if d >= len(used):
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        conflict = True
                        break
                if conflict:
                    for d in old_diffs:
                        used[d] = 1
                    continue
                
                for d in new_diffs:
                    used[d] = 1
                row[p_idx] = new_val
                cur_sum += new_val - old_val
                
                # Accept with probability if worse (simulated annealing lite)
                if cur_sum > best_sum:
                    # Revert if too bad? Or keep for exploration?
                    # For sum minimization, let's stick to hill climbing + random restarts
                    # instead of SA to keep it fast and deterministic in improvement.
                    pass 
                if cur_sum < best_sum:
                    best_sum = cur_sum
                    best_rows = [list(r) for r in rows]

        return best_rows, best_sum

    def local_search_max(rows, rng, max_iters=5000):
        """
        Local search to minimize the maximum entry.
        """
        rows = [list(r) for r in rows]
        n = len(rows)
        cur_max = max(max(r) for r in rows)
        best_max = cur_max
        best_rows = [list(r) for r in rows]
        
        max_val = cur_max
        used = bytearray(max_val + 2)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used[d] = 1

        for it in range(max_iters):
            if _time_left() < 5.0:
                break
            
            # Find the row with the current max element
            max_row_idx = -1
            max_val_in_row = -1
            for i in range(n):
                if rows[i][-1] > max_val_in_row:
                    max_val_in_row = rows[i][-1]
                    max_row_idx = i
            
            if max_row_idx == -1:
                break
                
            row = rows[max_row_idx]
            # Try to shrink the last element
            old_val = row[-1]
            lo = row[-2] + 1
            hi = old_val - 1
            if hi < lo:
                # Try to shrink an interior element of the max row to free up space?
                # Or just try to shift the max element down by 1 if possible
                new_val = old_val - 1
                if new_val <= lo:
                    # Try a random interior move in this row
                    p_idx = rng.randint(1, len(row) - 2)
                    old_v = row[p_idx]
                    lo_i = row[p_idx-1] + 1
                    hi_i = row[p_idx+1] - 1
                    if hi_i <= lo_i: continue
                    new_v = rng.randint(lo_i, hi_i)
                    if new_v == old_v: continue
                    
                    old_diffs = []
                    new_diffs = []
                    for j in range(len(row)):
                        if j == p_idx: continue
                        if j < p_idx:
                            old_diffs.append(old_v - row[j])
                            new_diffs.append(new_v - row[j])
                        else:
                            old_diffs.append(row[j] - old_v)
                            new_diffs.append(row[j] - new_v)
                    
                    for d in old_diffs:
                        used[d] = 0
                    conflict = False
                    for d in new_diffs:
                        if d >= len(used):
                            used.extend(bytearray(d - len(used) + 1))
                        if used[d]:
                            conflict = True
                            break
                    if conflict:
                        for d in old_diffs:
                            used[d] = 1
                        continue
                    for d in new_diffs:
                        used[d] = 1
                    row[p_idx] = new_v
                    # Max might not change, but structure did.
                    continue

            # Attempt to shrink last element
            new_val = rng.randint(lo, hi)
            old_diffs = [old_val - a for a in row[:-1]]
            new_diffs = [new_val - a for a in row[:-1]]
            
            for d in old_diffs:
                used[d] = 0
            conflict = False
            for d in new_diffs:
                if d >= len(used):
                    used.extend(bytearray(d - len(used) + 1))
                if used[d]:
                    conflict = True
                    break
            if conflict:
                for d in old_diffs:
                    used[d] = 1
                continue
            
            for d in new_diffs:
                used[d] = 1
            row[-1] = new_val
            new_max = max(max(r) for r in rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in rows]

        return best_rows, best_max

    # Phase 0: Initialization
    rng = random.Random(42)
    baseline = greedy_construct(n_rows, k, rng)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)
    best_sum = sum(sum(r) for r in baseline)

    # Phase 1: Sum Minimization with Restarts
    # We run several independent local searches from slightly perturbed starts
    restarts = 10
    for i in range(restarts):
        if _time_left() < 20.0:
            break
        
        # Perturb the baseline
        current = [list(r) for r in baseline]
        # Add small random shifts to interior elements to escape greedy traps
        for _ in range(20):
            r_idx = rng.randrange(n_rows)
            p_idx = rng.randint(1, k)
            old = current[r_idx][p_idx]
            lo = current[r_idx][p_idx-1] + 1
            hi = current[r_idx][p_idx+1] - 1 if p_idx < k else old + 10
            if hi > lo:
                new = rng.randint(lo, min(hi, old + 5))
                # Check validity quickly? Or just let local search fix it?
                # Local search assumes valid start. So we must ensure perturbation is valid.
                # This is hard. Let's stick to valid perturbations.
                # Compute diffs for this move
                row = current[r_idx]
                old_diffs = []
                new_diffs = []
                for j in range(len(row)):
                    if j == p_idx: continue
                    if j < p_idx:
                        old_diffs.append(old - row[j])
                        new_diffs.append(new - row[j])
                    else:
                        old_diffs.append(row[j] - old)
                        new_diffs.append(row[j] - new)
                
                # Check against all other diffs
                all_other_diffs = set()
                for r in current:
                    if r is row: continue
                    for x in range(len(r)):
                        for y in range(x+1, len(r)):
                            all_other_diffs.add(r[y]-r[x])
                # And diffs within row not involving p_idx
                for x in range(len(row)):
                    for y in range(x+1, len(row)):
                        if x == p_idx or y == p_idx: continue
                        all_other_diffs.add(row[y]-row[x])
                
                if not set(new_diffs).intersection(all_other_diffs):
                    row[p_idx] = new

        # Run sum minimization
        improved, new_sum = local_search_sum(current, rng, max_iters=2000)
        if new_sum < best_sum:
            best_sum = new_sum
            # Now refine max for this good sum solution
            refined, refined_max = local_search_max(improved, rng, max_iters=2000)
            if refined_max < best_max:
                best_max = refined_max
                best_rows = refined

    # Phase 2: Final Tightening on the best found
    if _time_left() > 10.0:
        final_rows, final_max = local_search_max(best_rows, rng, max_iters=5000)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows

    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k, rng)

    return best_rows
