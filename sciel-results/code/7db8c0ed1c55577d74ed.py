# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a Rebuild-Based Local Search (structural change from SA) to navigate
    the discrete landscape, treating rows as variables in a tomography-like
    constraint satisfaction problem.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
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
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
        best_max = max(max(r) for r in best_rows)

    def get_cost(rows):
        """
        Cost function: Primary is max entry, Secondary is number of collisions.
        We want to minimize max entry. If max entry is same, we prefer fewer collisions.
        """
        max_val = 0
        diff_count = {}
        for row in rows:
            if row:
                max_val = max(max_val, row[-1])
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        
        collisions = 0
        for cnt in diff_count.values():
            if cnt > 1:
                collisions += cnt - 1
        
        # We prioritize valid solutions (collisions=0) with low max.
        # If invalid, we prioritize low collisions, then low max.
        # To guide search towards validity first if we are far from it,
        # but the greedy start is valid, so we likely stay near valid.
        # Let's use a lexicographic cost: (max_val, collisions)
        # But we want to reduce max_val.
        # If we have collisions, we should fix them first?
        # Actually, the task requires distinct differences.
        # So any solution with collisions is invalid.
        # We should minimize collisions first to find a valid solution, then minimize max.
        # However, the greedy solution is valid.
        # So we can just minimize max, but we must ensure we don't accept invalid states
        # unless they have fewer collisions than the current valid state?
        # No, if we accept an invalid state, we might get stuck.
        # Let's define cost as:
        # If collisions > 0: cost = collisions * 10000 + max_val
        # If collisions == 0: cost = max_val
        # This ensures we always prefer valid solutions over invalid ones (if max_val is reasonable).
        # But if max_val is huge in valid, and small in invalid, we might jump to invalid.
        # Since we start valid, and we want to keep it valid, we should penalize collisions heavily.
        
        if collisions > 0:
            return collisions * 10000 + max_val
        else:
            return max_val

    def local_search(n_rows, k, initial_rows):
        rng = random.Random(42)
        rows = [list(r) for r in initial_rows]
        current_cost = get_cost(rows)
        best_rows = [list(r) for r in rows]
        best_cost = current_cost
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 2.0:
            if no_improve > max_no_improve:
                # Structural change: Rebuild a random row to escape local minima
                # This is the "rebuild-based" strategy from the knowledge.
                ri = rng.randrange(n_rows)
                # Rebuild row ri greedily but with some randomness or just reset to greedy
                # Let's try to rebuild it to be smaller.
                # We can remove row ri, and try to insert a new one that fits.
                # Or just reset it to a greedy row and see if it improves?
                # Better: Randomly perturb the row significantly.
                
                # Strategy: Reset row ri to a random valid-like structure
                # or just try to minimize its max entry given the other rows.
                
                # Let's try a "rebuild" where we clear the row and greedily fill it,
                # but we allow it to be "looser" initially? No, greedy is tight.
                
                # Alternative Rebuild:
                # 1. Identify the row with the largest max.
                # 2. Try to reconstruct it to be smaller.
                
                target_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
                
                # Remove target row from consideration for differences
                other_rows = [r for i, r in enumerate(rows) if i != target_row_idx]
                used_diffs = set()
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            used_diffs.add(r[j] - r[i])
                
                # Rebuild target_row
                new_row = [0]
                x = 0
                # Try to find a compact row
                # We can use a local search on just this row?
                # Or just greedy. Greedy is a good baseline.
                # Let's try a few random restarts for this row.
                best_new_row = None
                best_new_row_max = float('inf')
                
                for _ in range(5):
                    temp_row = [0]
                    curr_x = 0
                    while len(temp_row) < k + 1:
                        curr_x += 1
                        new_diffs = {curr_x - a for a in temp_row}
                        if not (new_diffs & used_diffs) and not (new_diffs & {a for a in temp_row}): # self diffs are always distinct if increasing
                            # Check self diffs distinct? Yes, strictly increasing.
                            # Check against used_diffs
                            if not (new_diffs & used_diffs):
                                temp_row.append(curr_x)
                                # We don't update used_diffs yet because we are testing
                                # We need to check if this row is valid internally? Yes.
                                # But we also need to ensure we can finish.
                                # Greedy is fine.
                                pass
                            else:
                                # Greedy failed? No, greedy always succeeds if we go high enough.
                                # Wait, the greedy loop above doesn't check "can finish".
                                # It just appends.
                                # If it gets stuck? It won't, x increases indefinitely.
                                # But we want compact.
                                # The standard greedy is:
                                # for x in 1..: if diffs valid, add.
                                # This is what we did in greedy_construct.
                                # So let's just use that.
                                break
                    # The loop above was flawed. Let's just use the greedy function logic.
                    # Re-greedy for this row.
                    temp_row = [0]
                    curr_x = 0
                    while len(temp_row) < k + 1:
                        curr_x += 1
                        new_diffs = {curr_x - a for a in temp_row}
                        if not (new_diffs & used_diffs):
                            temp_row.append(curr_x)
                    if temp_row[-1] < best_new_row_max:
                        best_new_row_max = temp_row[-1]
                        best_new_row = temp_row
                
                if best_new_row:
                    rows[target_row_idx] = best_new_row
                    new_cost = get_cost(rows)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        best_rows = [list(r) for r in rows]
                        best_cost = current_cost
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
                continue
            
            # Normal move: Perturb a random element in a random row
            ri = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old_val = rows[ri][pos]
            
            # Try a small change
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
            
            valid_move = True
            if new_val <= rows[ri][pos - 1]:
                valid_move = False
            if pos < k and new_val >= rows[ri][pos + 1]:
                valid_move = False
            if new_val <= 0:
                valid_move = False
            
            if not valid_move:
                no_improve += 1
                continue
            
            rows[ri][pos] = new_val
            new_cost = get_cost(rows)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in rows]
                no_improve = 0
            else:
                # Reject
                rows[ri][pos] = old_val
                no_improve += 1
        
        return best_rows, best_cost

    ls_rows, ls_cost = local_search(n_rows, k, best_rows)
    
    # Validate the result from local search
    if validate(ls_rows, n_rows, k):
        ls_max = max(max(r) for r in ls_rows)
        if ls_max < best_max:
            best_rows = ls_rows
            best_max = ls_max
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
