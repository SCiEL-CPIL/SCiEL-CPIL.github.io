import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_remaining():
        return deadline - time.time()

    # --- Core validation and cost ---
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def calculate_scope(rows):
        return max(max(r) for r in rows)

    def is_valid_and_scope(rows):
        """
        Returns (is_valid, scope, conflicts_count, diffs_set)
        """
        used_diffs = set()
        valid = True
        scope = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        valid = False
                        conflicts += 1
                    else:
                        used_diffs.add(d)
            if max(row) > scope:
                scope = max(row)
        return valid, scope, conflicts, used_diffs

    def get_cost(rows):
        """
        Cost = scope if valid, else scope + penalty * conflicts
        """
        valid, scope, conflicts, _ = is_valid_and_scope(rows)
        if valid:
            return scope
        else:
            return scope + 100 * conflicts

    # --- Initialization Strategies ---
    def greedy_construct():
        """Deterministic greedy: always pick the smallest valid next element."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows

    def greedy_random_construct(perturb=0):
        """
        Randomized greedy: pick from a small window of valid candidates.
        This creates diverse initial populations.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Look ahead for a few valid candidates
                candidates = []
                search_limit = x + 10 + perturb
                while x < search_limit:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append((x, new_diffs))
                    if len(candidates) >= 5:
                        break
                
                if not candidates:
                    # If no candidates in window, expand search
                    x += 1
                    continue

                # Pick randomly from candidates
                chosen_val, chosen_diffs = random.choice(candidates)
                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                
                # Skip x to chosen_val + 1 to maintain speed and order
                x = chosen_val
        
        # Verify validity, if invalid (due to weird random choices), fallback to greedy for this row?
        # Actually, greedy_random as implemented above is strictly sequential and ensures validity by construction
        # because we only pick x if it doesn't conflict with used_diffs.
        return rows

    # --- Local Search Move: Efficient Element Shift ---
    def try_shift_element(rows, ri, ci, delta, used_diffs):
        """
        Attempt to shift element at (ri, ci) by delta.
        Returns (success, new_used_diffs) or (False, None).
        """
        row = rows[ri]
        old_val = row[ci]
        new_val = old_val + delta
        
        # Bounds check
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None
        
        # Collect old diffs involving ci
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])
            
        # Remove old diffs from used set
        temp_used = used_diffs - old_diffs
        
        # Check new diffs
        new_diffs = set()
        for i in range(ci):
            d = new_val - row[i]
            if d in temp_used:
                return False, None
            new_diffs.add(d)
        for j in range(ci + 1, len(row)):
            d = row[j] - new_val
            if d in temp_used:
                return False, None
            new_diffs.add(d)
            
        # Success
        new_used = temp_used | new_diffs
        return True, new_used

    def local_search_reduce_max(rows, used_diffs, max_iters=5000):
        """
        Targeted local search to reduce the maximum entry.
        Focuses on the row with the largest last element and tries to shift elements down.
        """
        rows = [row[:] for row in rows]
        used_diffs = set(used_diffs)
        current_scope = max(r[-1] for r in rows)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            
            # Try to reduce elements in this row, starting from the end (most impact on scope)
            improved = False
            for ci in range(len(rows[max_row_idx]) - 1, 0, -1):
                current_val = rows[max_row_idx][ci]
                # Try small negative shifts
                for delta in range(-1, -5, -1):
                    success, new_used = try_shift_element(rows, max_row_idx, ci, delta, used_diffs)
                    if success:
                        rows[max_row_idx][ci] += delta
                        used_diffs = new_used
                        new_scope = max(r[-1] for r in rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                        improved = True
                        break
                if improved:
                    break
            
            # If no improvement in max row, try other rows to free up space
            if not improved:
                for ri in range(n_rows):
                    if ri == max_row_idx:
                        continue
                    for ci in range(len(rows[ri]) - 1, 0, -1):
                        current_val = rows[ri][ci]
                        for delta in range(-1, -5, -1):
                            success, new_used = try_shift_element(rows, ri, ci, delta, used_diffs)
                            if success:
                                rows[ri][ci] += delta
                                used_diffs = new_used
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
            if not improved:
                # Stuck, break or randomize? For now, break to let SA take over or restart
                break
                
        return rows, used_diffs, current_scope

    def simulated_annealing(rows, used_diffs, time_limit):
        """
        Simulated Annealing to escape local minima and further reduce scope.
        """
        rows = [row[:] for row in rows]
        used_diffs = set(used_diffs)
        current_scope = max(r[-1] for r in rows)
        best_scope = current_scope
        best_rows = [row[:] for row in rows]
        
        T = 20.0
        alpha = 0.995
        min_T = 0.1
        
        while time.time() < time_limit:
            # Pick random row and index
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, k)
            
            # Generate random delta
            delta_range = random.randint(1, 10)
            delta = random.randint(-delta_range, delta_range)
            
            if delta == 0:
                continue
                
            success, new_used = try_shift_element(rows, r_idx, i_idx, delta, used_diffs)
            if not success:
                continue
            
            # Calculate cost change
            # Old cost approx: current_scope (assuming valid)
            # New cost:
            new_scope = max(r[-1] for r in rows)
            
            # We need to check validity implicitly via the used_diffs set update logic in try_shift_element
            # If try_shift_element succeeded, the move is valid with respect to differences.
            # However, we must ensure the row remains strictly increasing (handled by bounds in try_shift_element)
            
            delta_cost = new_scope - current_scope
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                rows[r_idx][i_idx] += delta
                used_diffs = new_used
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [row[:] for row in rows]
            else:
                # Reject (no state change needed as we didn't modify rows in place yet in a committed way? 
                # Wait, try_shift_element returns new_used but doesn't modify rows.
                # So we just discard new_used.
                pass
            
            T *= alpha
            if T < min_T:
                T = 10.0 # Reheat
            
        return best_rows, best_scope

    # --- Main Execution ---
    
    # 1. Generate a diverse set of initial candidates
    candidates = []
    
    # Deterministic greedy
    g1 = greedy_construct()
    valid, scope, _, diffs = is_valid_and_scope(g1)
    if valid:
        candidates.append((scope, g1, diffs))
    
    # Multiple randomized greedy restarts
    for i in range(20):
        if time.time() > deadline - 30:
            break
        # Increase perturb slightly to explore wider
        pert = i // 5
        g = greedy_random_construct(perturb=pert)
        valid, scope, _, diffs = is_valid_and_scope(g)
        if valid:
            candidates.append((scope, g, diffs))
    
    if not candidates:
        # Fallback
        return greedy_construct()
        
    # Sort candidates by scope
    candidates.sort(key=lambda x: x[0])
    
    best_scope = candidates[0][0]
    best_rows = candidates[0][1]
    best_diffs = candidates[0][2]
    
    # 2. Refine the top candidates using local search and SA
    # We will spend time refining the top 3 candidates
    
    for idx, (scope, rows, diffs) in enumerate(candidates[:3]):
        if time.time() > deadline - 5:
            break
            
        # Phase A: Targeted Local Search to reduce max
        # Allocate time based on remaining budget
        alloc_time = min(10.0, (deadline - time.time()) / 2.0)
        if alloc_time < 1:
            continue
            
        ls_rows, ls_diffs, ls_scope = local_search_reduce_max(rows, diffs, max_iters=int(alloc_time * 500))
        
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows
            best_diffs = ls_diffs
            
        # Phase B: Simulated Annealing
        # Use remaining time for this candidate or a fixed chunk
        sa_time_limit = time.time() + min(5.0, (deadline - time.time()) / 3.0)
        if sa_time_limit > deadline - 2:
            continue
            
        sa_rows, sa_scope = simulated_annealing(ls_rows, ls_diffs, sa_time_limit)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
            # Recalculate diffs for best_rows just in case
            _, _, _, best_diffs = is_valid_and_scope(best_rows)
            
        # If we reached the theoretical lower bound, we can stop early
        if best_scope <= 112:
            break

    return best_rows
