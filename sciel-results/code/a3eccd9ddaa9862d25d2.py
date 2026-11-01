import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 135.0  # Safety margin

    # Problem specific parameters for (4, 7)
    # n_rows = 4, k = 7 implies 8 points per row
    # Total diffs = 4 * (7*8/2) = 112
    # Lower bound for scope m: m(m+1)/2 >= 112 => m >= 15 (120/2=60, no, 15*16/2=120 >= 112)
    # Actually, the counting bound is on the number of differences.
    # m=15 gives 120 differences. We need 112. So m=15 is the theoretical min.
    # Best known is 134? No, the prompt says "best known set has largest entry 134" but then 
    # "simple counting bound is 112". Wait, the score 1.88 corresponds to scope ~114.
    # 215 / 114 = 1.885.
    # So we are searching for scope < 114.

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_init():
        """
        Greedy construction:
        Place points sequentially to minimize the next available position.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We want to place k more points
            for _ in range(k):
                # Try to find the smallest x > row[-1] such that all diffs are new
                x = row[-1] + 1
                while True:
                    # Check diffs with all previous points in this row
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        # Check if x itself creates any conflict? 
                        # No, just the differences matter.
                        # Add diffs to used
                        for a in row:
                            used_diffs.add(x - a)
                        row.append(x)
                        break
                    x += 1
            rows.append(row)
        return rows

    # Initial solution
    best_rows = greedy_init()
    best_scope = get_scope(best_rows)
    
    # Pre-calculate the set of all used differences for the best solution
    # We will maintain this set globally for fast validation
    def build_diff_set(rows):
        s = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    s.add(r[j] - r[i])
        return s

    global_used = build_diff_set(best_rows)

    rng = random.Random(42)

    def try_move(cur_rows, ri, pos, new_val):
        """
        Attempt to move cur_rows[ri][pos] to new_val.
        Returns (success, new_scope).
        Modifies cur_rows and global_used in place if successful.
        """
        row = cur_rows[ri]
        old_val = row[pos]
        if old_val == new_val:
            return False, get_scope(cur_rows)
        
        # Check bounds
        if pos > 0 and new_val <= row[pos-1]:
            return False, get_scope(cur_rows)
        if pos < len(row)-1 and new_val >= row[pos+1]:
            return False, get_scope(cur_rows)
        
        # Calculate old diffs to remove
        old_diffs = []
        for i in range(pos):
            d = old_val - row[i]
            old_diffs.append(d)
        for j in range(pos+1, len(row)):
            d = row[j] - old_val
            old_diffs.append(d)
        
        # Calculate new diffs to add
        new_diffs = []
        for i in range(pos):
            d = new_val - row[i]
            new_diffs.append(d)
        for j in range(pos+1, len(row)):
            d = row[j] - new_val
            new_diffs.append(d)
            
        # Check for conflicts
        # A diff is invalid if it's in global_used AND not in old_diffs (which are being removed)
        # Note: old_diffs might have duplicates? No, row is strictly increasing, so diffs are unique within the row.
        # But we must ensure we don't count a diff that is being removed as a conflict for a new diff.
        
        # Optimization: 
        # 1. Remove old_diffs from global_used
        # 2. Check if any new_diff is in global_used
        # 3. If yes, restore old_diffs and return False
        # 4. If no, add new_diffs to global_used and return True
        
        # To avoid modifying and restoring on failure (which is slow), we can check first:
        # Conflict exists if:
        # - d in new_diffs AND d in global_used AND d not in old_diffs
        
        old_diff_set = set(old_diffs)
        
        for d in new_diffs:
            if d in global_used and d not in old_diff_set:
                # Conflict found
                return False, get_scope(cur_rows)
        
        # No conflict. Apply move.
        # Remove old
        for d in old_diffs:
            global_used.discard(d)
        # Add new
        for d in new_diffs:
            global_used.add(d)
        
        # Update row
        row[pos] = new_val
        
        # Calculate new scope
        # Since we only changed one element, the new max is max(old_max, new_val) if new_val > old_max?
        # Actually, if we moved the max down, scope might decrease.
        # If we moved an inner element, scope might increase if new_val > current_max.
        # Simplest: recalculate max of the row, then max of all rows.
        # But we can track it.
        # For simplicity and safety, just get_scope. It's O(n*k) which is small (32 ops).
        new_scope = get_scope(cur_rows)
        
        return True, new_scope

    def revert_move(cur_rows, ri, pos, old_val, new_val):
        """Revert a successful move."""
        row = cur_rows[ri]
        
        # Revert diffs
        # Current diffs (new)
        new_diffs = []
        for i in range(pos):
            new_diffs.append(new_val - row[i])
        for j in range(pos+1, len(row)):
            new_diffs.append(row[j] - new_val)
            
        # Old diffs (to restore)
        old_diffs = []
        for i in range(pos):
            old_diffs.append(old_val - row[i])
        for j in range(pos+1, len(row)):
            old_diffs.append(row[j] - old_val)
            
        for d in new_diffs:
            global_used.discard(d)
        for d in old_diffs:
            global_used.add(d)
            
        row[pos] = old_val

    def targeted_descent(rows, budget):
        """
        Try to reduce scope by shrinking the largest elements.
        """
        start = time.time()
        cur = [r[:] for r in rows]
        # Rebuild global_used for cur
        # We need to sync global_used with cur.
        # Easiest: rebuild from scratch.
        global global_used
        old_global = global_used
        global_used = build_diff_set(cur)
        
        cur_scope = get_scope(cur)
        improved = False
        
        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1
            
            if old_val <= lower:
                continue
                
            # Try to move pos to a smaller value
            # Try values from lower to old_val-1
            # We want the smallest valid value to maximize reduction?
            # Or just any valid value < old_val?
            # If we find a valid value v < old_val, scope becomes max(max(other rows), v).
            # So we want the smallest v.
            
            found = False
            for v in range(lower, old_val):
                if try_move(cur, r, pos, v):
                    # Success
                    new_scope = get_scope(cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        found = True
                        break
                    else:
                        # Revert
                        revert_move(cur, r, pos, old_val, v)
                        # If scope didn't improve, maybe a slightly larger v is better?
                        # No, if v < old_val, max(cur[r]) becomes v (assuming v is the new max of row r).
                        # If v < max(other rows), scope doesn't change.
                        # If v < old_val but v >= max(other rows), scope becomes v.
                        # So any valid v < old_val is good if it reduces the global max.
                        # If it doesn't reduce global max, it's neutral.
                        # We break on first improvement.
                # If try_move fails, it reverts itself? No, try_move only modifies on success.
                # So on failure, state is unchanged.
                
            if not found:
                # Try moving other elements in the max row?
                # Or just break if no improvement in this round
                # To avoid getting stuck, try a random move in this row
                pos2 = rng.randint(1, len(row)-2)
                old2 = row[pos2]
                lo2 = row[pos2-1] + 1
                hi2 = row[pos2+1] - 1
                if hi2 > lo2:
                    v = rng.randint(lo2, hi2)
                    if try_move(cur, r, pos2, v):
                        new_scope = get_scope(cur)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            improved = True
                        else:
                            revert_move(cur, r, pos2, old2, v)
                    if not improved:
                        break
                else:
                    break
        
        # Restore global_used to match best_rows (or whatever was passed)
        # Actually, targeted_descent should return the improved rows.
        # We modified global_used to match cur.
        # We should restore global_used to the state of the input 'rows' before returning?
        # Or just return cur and let the caller rebuild?
        # Rebuilding is O(N^2). N=32. 1024 ops. Cheap.
        # But we need to ensure global_used is consistent for the next call.
        # Let's restore global_used to the state of 'rows' (the input).
        global_used = build_diff_set(rows)
        
        if improved:
            return cur, cur_scope
        return None, None

    # Main Search Loop
    max_restarts = 100
    steps_per_restart = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize current state
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri])-1)
                lo = cur[ri][pos-1] + 1
                hi = cur[ri][pos+1] - 1 if pos < len(cur[ri])-1 else best_scope + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)
                    
        # Sync global_used with cur
        global_used = build_diff_set(cur)
        cur_scope = get_scope(cur)
        
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]
            # Note: global_used is now synced with cur, which is best_rows.
            
        # Annealing
        T = 1.0
        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break
                
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri])-1)
            lo = cur[ri][pos-1] + 1
            hi = cur[ri][pos+1] - 1 if pos < len(cur[ri])-1 else best_scope + 20
            if hi <= lo:
                continue
                
            new_val = rng.randint(lo, hi)
            
            old_val = cur[ri][pos]
            
            if try_move(cur, ri, pos, new_val):
                new_scope = get_scope(cur)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                    
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    revert_move(cur, ri, pos, old_val, new_val)
            # If try_move fails, state is unchanged.
            
            T *= 0.99995
            
        # Targeted Descent
        if time.time() < deadline - 3.0:
            desc_rows, desc_scope = targeted_descent(cur, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                # Sync global_used to best_rows for next iteration?
                # targeted_descent restored global_used to 'cur' (input to td).
                # But best_rows is now desc_rows.
                # We need global_used to match cur for the next SA loop.
                # The next loop starts by setting cur from best_rows.
                # So we should ensure global_used matches cur at the start of the loop.
                # The loop does: cur = ...; global_used = build_diff_set(cur).
                # So it's fine.

    return best_rows

