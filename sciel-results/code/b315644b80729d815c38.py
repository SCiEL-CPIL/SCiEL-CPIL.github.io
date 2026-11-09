"""
Difference Triangle Set (7,5) - Combined Strategy
Combines fast interleaved greedy initialization with Simulated Annealing
row-replacement refinement and final deterministic polish.
"""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: 150s limit, stop at 142s to allow for cleanup/return
    deadline = start_time + 142.0

    def elapsed():
        return time.time() - start_time

    def scope_of(rows):
        if not rows:
            return float('inf')
        return max(max(r) for r in rows)

    def get_diffs(row):
        """Calculate all unique positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ai)
        return diffs

    def build_interleaved(n_rows, k, max_scope=None):
        """
        Greedily build rows interleaved: extend the row with the smallest
        current maximum. This keeps rows balanced and compact.
        """
        rows = [[0] for _ in range(n_rows)]
        used = set()
        max_x = max_scope if max_scope is not None else 200

        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            # Pick the row with the smallest current max
            i = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[i]
            x = row[-1] + 1
            found = False
            while x <= max_x:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                return None

        # Verify completion
        for r in rows:
            if len(r) != k + 1:
                return None
        return rows

    def build_sequential(n_rows, k):
        """
        Simple sequential greedy. Builds one row completely before starting the next.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows

    # --- Phase 1: Initialization ---
    # Try interleaved first, it usually yields better compactness
    best_rows = build_interleaved(n_rows, k)
    if best_rows is None:
        best_rows = build_sequential(n_rows, k)
    
    if best_rows is None:
        # Fallback: should not happen, but just in case
        return [[0, 1, 2, 3, 4, 5]] * n_rows

    best_scope = scope_of(best_rows)

    # Try a few random restarts of interleaved if time permits and initial scope is high
    # For (7,5), interleaved is usually very good. Let's just do one random variation
    if elapsed() < 5.0:
        rng_init = random.Random(123)
        # Randomized interleaved: pick random candidate instead of min
        rows_rand = [[0] for _ in range(n_rows)]
        used_rand = set()
        success = True
        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows_rand[i]) < k + 1]
            if not candidates:
                success = False
                break
            # Weight towards smaller max, but add randomness
            # Simple: pick random from top 2 smallest
            candidates.sort(key=lambda idx: rows_rand[idx][-1])
            if len(candidates) > 1 and rng_init.random() < 0.5:
                i = rng_init.choice(candidates[:2])
            else:
                i = candidates[0]
            
            row = rows_rand[i]
            x = row[-1] + 1
            found = False
            while x < 200:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_rand):
                    row.append(x)
                    used_rand |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                success = False
                break
        
        if success and all(len(r) == k + 1 for r in rows_rand):
            sc = scope_of(rows_rand)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows_rand

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 2: Simulated Annealing with Row Replacement ---
    # This is the structural improvement from the inspiration.
    # We replace entire rows to allow large structural changes.
    
    def replace_row(rows, ri, used, k, max_mark, rnd, randomize_prob=0.1):
        """
        Try to replace row ri with a new row that has a smaller scope (max_mark)
        and does not conflict with 'used' differences.
        """
        old_diffs = get_diffs(rows[ri])
        used_other = used - old_diffs
        
        # We want the new row to have max < current row max, or at least < global best scope
        # Let's target max_mark
        for attempt in range(20):
            row = [0]
            local = set()
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                
                # Randomization to escape local traps in row construction
                if rnd.random() < randomize_prob and x < max_mark - 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_other or dv in local:
                        bad = True
                        break
                if not bad:
                    row.append(x)
                    local.update({x - a for a in row[:-1]})
            
            if success and len(row) == k + 1:
                new_diffs = get_diffs(row)
                if not (new_diffs & used_other):
                    return row, used_other | new_diffs
        return None, None

    def simulated_annealing(rows, n_rows, k, time_limit, rnd):
        current = [r[:] for r in rows]
        best = [r[:] for r in rows]
        
        used = set()
        for r in current:
            used |= get_diffs(r)
            
        cur_scope = scope_of(current)
        best_scope = cur_scope
        
        deadline_local = time.time() + time_limit
        T = 50.0
        cooling = 0.999
        
        while time.time() < deadline_local and T > 0.1:
            # Pick a random row to replace
            ri = rnd.randint(0, n_rows - 1)
            old_scope = current[ri][-1]
            
            # Try to find a replacement with scope <= old_scope (or slightly less)
            # Using old_scope as max_mark ensures we don't increase scope for this row
            # We want to reduce global scope, so if this row is the max, we must reduce it.
            # If it's not the max, we can try to reduce it or keep it same.
            # Strategy: Try to replace with max_mark = old_scope - 1 if possible, 
            # otherwise old_scope.
            
            # First try to strictly reduce this row's scope
            new_row, new_used = replace_row(current, ri, used, k, old_scope - 1, rnd)
            
            if new_row is None:
                # Try to replace with same scope but different structure (to help others?)
                # Or just skip. Replacing with same scope rarely helps global scope
                # unless it frees up differences for others.
                # Let's try replacing with same scope to shuffle differences
                if rnd.random() < 0.3:
                    new_row, new_used = replace_row(current, ri, used, k, old_scope, rnd, randomize_prob=0.2)
            
            if new_row is not None:
                new_scope = scope_of(current) # Wait, need to calculate new global scope
                # Calculate new global scope
                # We know current[ri] changes to new_row
                # new_scope = max(max(r) for r in current if r is not current[ri]) vs new_row[-1]
                
                # Update state
                current[ri] = new_row
                used = new_used
                
                # Calculate actual new scope
                max_other = max(r[-1] for i, r in enumerate(current) if i != ri) if n_rows > 1 else 0
                actual_new_scope = max(max_other, new_row[-1])
                
                delta = actual_new_scope - cur_scope
                
                # Acceptance criterion
                if delta < 0 or (T > 0 and rnd.random() < pow(2.71828, -delta / T)):
                    cur_scope = actual_new_scope
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best = [r[:] for r in current]
                
                T *= cooling
        
        return best, best_scope

    if elapsed() < deadline - 10.0:
        rng_sa = random.Random(42)
        time_budget = deadline - time.time() - 5.0
        improved_rows, improved_scope = simulated_annealing(best_rows, n_rows, k, time_budget, rng_sa)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 3: Final Deterministic Polish ---
    # Try to reduce individual elements by shifting them down.
    # This cleans up any "wasted" space left by SA.
    
    def final_polish(rows, n_rows, k):
        rows = [r[:] for r in rows]
        used = set()
        for r in rows:
            used |= get_diffs(r)
        
        best_scope = scope_of(rows)
        improved = True
        while improved:
            improved = False
            for ri in range(n_rows):
                row = rows[ri]
                for pi in range(1, len(row)):
                    old_val = row[pi]
                    # Try to move row[pi] to a smaller value
                    for new_val in range(old_val - 1, 0, -1):
                        if new_val <= row[pi-1]:
                            continue
                        if pi < len(row) - 1 and new_val >= row[pi+1]:
                            continue
                        
                        # Calculate diff changes
                        old_diffs = set()
                        new_diffs = set()
                        
                        # Diffs involving pi and elements before pi
                        for i in range(pi):
                            old_diffs.add(old_val - row[i])
                            new_diffs.add(new_val - row[i])
                        
                        # Diffs involving pi and elements after pi
                        for j in range(pi + 1, len(row)):
                            old_diffs.add(row[j] - old_val)
                            new_diffs.add(row[j] - new_val)
                        
                        # Check validity
                        remaining_used = used - old_diffs
                        if not (new_diffs & remaining_used):
                            # Valid move
                            used = remaining_used | new_diffs
                            row[pi] = new_val
                            new_scope = scope_of(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                            improved = True
                            break # Restart scan for this row/point? 
                            # Actually, just break the new_val loop and continue to next pi/row
                            
        return rows, best_scope

    if elapsed() < deadline - 1.0:
        polished_rows, polished_scope = final_polish(best_rows, n_rows, k)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_rows = polished_rows

    return best_rows

if __name__ == "__main__":
    # Quick self-test
    rows = construct(7, 5)
    print(f"Scope: {max(max(r) for r in rows)}")
    print(rows)
