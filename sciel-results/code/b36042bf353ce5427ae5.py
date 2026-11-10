import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using randomized greedy restarts
    and a targeted local search that aggressively minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid DTS. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            if not row or row[0] != 0:
                return False, float('inf')
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            max_entry = max(max_entry, row[-1])
        return True, max_entry

    def compute_diffs(row):
        """Compute all positive differences in a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_initial(n_rows, k, rng, perturb=0):
        """
        Fast greedy initialization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if time.time() > deadline - 10:
                    return None
                
                # Determine search limit
                # We want to find a valid x quickly.
                # If perturb is high, we might skip some early candidates.
                search_limit = x + 40 + (perturb * 15)
                candidates = []
                
                while x < search_limit:
                    # Check if x is valid
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        candidates.append((x, new_diffs))
                    
                    x += 1
                    if time.time() > deadline - 10:
                        return None

                if not candidates:
                    # If no candidates found in window, expand window
                    x = search_limit
                    continue

                # Choose a candidate
                # To encourage diversity, we pick randomly from the top few
                if len(candidates) > 1 and rng.random() < 0.3:
                    top_n = min(5, len(candidates))
                    idx = rng.randint(0, top_n - 1)
                    chosen_val, chosen_diffs = candidates[idx]
                else:
                    # Greedy: pick the smallest
                    chosen_val, chosen_diffs = candidates[0]
                
                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def get_max_entry(rows):
        return max(row[-1] for row in rows)

    def try_move(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs, new_max) or (False, None, None).
        """
        row = rows[ri]
        current_val = row[ci]
        
        # Check bounds
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None, None
        
        if new_val == current_val:
            return False, None, None

        # Calculate old diffs involving (ri, ci)
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(current_val - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - current_val)
        
        # Remove old diffs from used
        temp_used = used_diffs - old_diffs
        
        # Calculate new diffs
        new_diffs = set()
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                return False, None, None
            new_diffs.add(d)
        for j in range(ci + 1, len(row)):
            d = row[j] - new_val
            if d <= 0 or d in temp_used:
                return False, None, None
            new_diffs.add(d)
        
        new_used = temp_used | new_diffs
        
        # Calculate new max
        new_row_max = max(row[-1] if ci != len(row)-1 else new_val, 
                          row[-1] if ci != len(row)-1 else row[-1])
        # Actually, just recompute max of the whole thing or update incrementally
        # For simplicity and correctness, let's just return the sets and let the caller check max
        # But we can be smart: if ci is the last element, the row max changes.
        
        return True, new_used, new_val

    def local_search_reduce(rows, n_rows, k, max_iters=50000):
        """
        Aggressive local search to reduce the maximum entry.
        Strategy:
        1. Find the row with the max last element.
        2. Try to reduce that last element by 1, 2, ... until valid.
        3. If stuck, perform a random "kick" (move an element in any row to a random valid position)
           that does not increase the current max, to escape local optima.
        """
        rows = [row[:] for row in rows]
        used_diffs = set()
        for row in rows:
            used_diffs.update(compute_diffs(row))
        
        best_score = get_max_entry(rows)
        best_rows = [r[:] for r in rows]
        
        T = 50.0
        T_min = 1.0
        cooling = 0.9995
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            current_max = get_max_entry(rows)
            
            # Find index of row with max last element
            max_row_idx = -1
            max_val = -1
            for i in range(n_rows):
                if rows[i][-1] > max_val:
                    max_val = rows[i][-1]
                    max_row_idx = i
            
            # 1. Try to reduce the max element
            reduced = False
            row = rows[max_row_idx]
            ci = len(row) - 1
            current_val = row[ci]
            
            # Try reducing by small amounts first (greedy reduction)
            for delta in range(1, min(10, current_val)):
                new_val = current_val - delta
                if new_val <= row[ci-1]:
                    break
                success, new_used, _ = try_move(rows, max_row_idx, ci, new_val, used_diffs)
                if success:
                    # Check if this move is valid globally (it is by construction of try_move)
                    # But we need to ensure it doesn't break the "increasing" property of other rows?
                    # try_move only checks local row consistency and global diff uniqueness.
                    # Since we only move one element in one row, and we check bounds, 
                    # the row remains sorted.
                    rows[max_row_idx][ci] = new_val
                    used_diffs = new_used
                    reduced = True
                    break
            
            if not reduced:
                # 2. If we couldn't reduce the max, try to "kick" to escape local optimum
                # Pick a random row and element (not the first one)
                ri = rng.randint(0, n_rows - 1)
                ci = rng.randint(1, len(rows[ri]) - 1)
                
                # Try a few random nearby moves
                moved = False
                for _ in range(10):
                    current_val = rows[ri][ci]
                    # Random offset
                    offset = rng.randint(-20, 20)
                    new_val = current_val + offset
                    
                    # Basic bounds check
                    if ci > 0 and new_val <= rows[ri][ci - 1]:
                        continue
                    if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
                        continue
                    if new_val <= 0:
                        continue
                    
                    # Only accept if it doesn't increase the global max significantly
                    # or if it's a small increase (SA style)
                    # For simplicity, let's just try to make a valid move.
                    # If the move reduces the max, great. If it keeps it same, good for escape.
                    # If it increases it, we might reject unless T is high.
                    
                    success, new_used, _ = try_move(rows, ri, ci, new_val, used_diffs)
                    if not success:
                        continue
                    
                    # Calculate new max
                    # We have to update the row to calculate max properly
                    old_val = rows[ri][ci]
                    rows[ri][ci] = new_val
                    new_max = get_max_entry(rows)
                    
                    delta_max = new_max - current_max
                    if delta_max <= 0 or (T > T_min and rng.random() < math.exp(-delta_max / T)):
                        # Accept move
                        used_diffs = new_used
                        moved = True
                        if new_max < best_score:
                            best_score = new_max
                            best_rows = [r[:] for r in rows]
                        break
                    else:
                        # Reject move, restore
                        rows[ri][ci] = old_val
                        
                if not moved:
                    # If no kick succeeded, cool down faster or break?
                    # Let's just cool down
                    T = max(T_min, T * 0.99)
            else:
                # If we reduced, update best
                new_max = get_max_entry(rows)
                if new_max < best_score:
                    best_score = new_max
                    best_rows = [r[:] for r in rows]
                # Cool down slowly as we are making progress
                T = max(T_min, T * 0.999)
            
            # Standard cooling
            T = max(T_min, T * cooling)

        return best_rows, best_score

    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # 1. Deterministic Greedy + Local Search
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            improved, improved_score = local_search_reduce(det_rows, n_rows, k, max_iters=20000)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # 2. Randomized Restarts
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        # Vary perturbation
        perturb = rng.randint(0, 10)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            # Run local search on this promising candidate
            # Allocate more time if we are close to the best
            iters = 30000
            improved, improved_score = local_search_reduce(rows, n_rows, k, max_iters=iters)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved
        else:
            # If not better than current best, only do local search if it's close
            # and we have time
            if max_entry <= best_score + 5 and time.time() < deadline - 30:
                improved, improved_score = local_search_reduce(rows, n_rows, k, max_iters=10000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_result = improved

    # 3. Final Polish
    if best_result is not None and time.time() < deadline - 10:
        final_improved, final_score = local_search_reduce(best_result, n_rows, k, max_iters=50000)
        if final_score < best_score:
            best_score = final_score
            best_result = final_improved

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
# EVOLVE-BLOCK-END
