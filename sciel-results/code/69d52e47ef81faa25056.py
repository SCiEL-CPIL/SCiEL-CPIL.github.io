import time
import random

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers)
    # We want to minimize M (max entry)
    
    start_time = time.time()
    # Safety margin for the 120s limit (evaluator kills at 150s, but we want to return well before).
    # Let's aim for 110s of work to be safe.
    deadline = start_time + 110.0

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Advanced Row Construction: Beam Search ----------
    def build_row_beam(used_other, beam_width=10, search_depth=2, max_x_limit=200):
        """
        Constructs a single row using a beam search strategy.
        Instead of picking the smallest valid x, it keeps the top 'beam_width' 
        partial rows based on a heuristic score (current max + potential).
        """
        # State: (row_list, current_max_diff_set, last_element)
        # We start with [0]
        beam = [([0], set(), 0)]
        
        for step in range(1, k + 1):
            next_beam = []
            for row, diffs, last_x in beam:
                # Find valid candidates for the next element
                candidates = []
                # We look ahead a bit further than the current last_x
                # The gap can be large, so we scan up to last_x + 50 or so
                # But we must also ensure we don't exceed a reasonable bound
                # If we are stuck, we might need to go higher.
                # However, for beam search, we want to find *good* (small) solutions.
                # If no small solution exists, the beam will die.
                
                # Optimization: We can estimate the minimum possible next value.
                # It must be > last_x.
                # It must not create conflicts with 'used_other' or 'diffs'.
                
                # We scan a window. If the window is too small, we might miss valid small numbers.
                # If the window is too large, it's slow.
                # Let's scan up to last_x + 60.
                limit = last_x + 60
                
                for trial_x in range(last_x + 1, limit + 1):
                    new_diffs = {trial_x - a for a in row}
                    if not (new_diffs & used_other) and not (new_diffs & diffs):
                        # Heuristic score: 
                        # 1. Prefer smaller trial_x
                        # 2. Prefer candidates that leave more "room" (fewer small diffs created?)
                        # Simple score: trial_x
                        candidates.append((trial_x, trial_x, new_diffs))
                
                if not candidates:
                    # If no candidate found in this window, we might need to expand
                    # But for beam search efficiency, we might just drop this branch
                    # or try a much larger jump. 
                    # Let's try a larger jump if we are early in the row.
                    if step < 3:
                         for trial_x in range(limit + 1, limit + 100):
                             new_diffs = {trial_x - a for a in row}
                             if not (new_diffs & used_other) and not (new_diffs & diffs):
                                 candidates.append((trial_x, trial_x, new_diffs))
                                 if len(candidates) > 5: break
                    if not candidates:
                        continue # Dead end for this branch

                # Sort candidates by score (smallest x first)
                candidates.sort(key=lambda c: c[0])
                
                # Add top candidates from this branch to next_beam
                # We want to keep diversity, so we take the top few from each branch
                for score, tx, nd in candidates[:3]:
                    new_row = row + [tx]
                    new_diffs = diffs | nd
                    next_beam.append((new_row, new_diffs, tx))
            
            if not next_beam:
                # If beam is empty, we failed to construct a row with this strategy
                # This can happen if used_other is very restrictive.
                # Fallback: Greedy
                return greedy_row_fallback(used_other)
            
            # Prune next_beam to beam_width
            # Sort by the last element (current max of the row)
            next_beam.sort(key=lambda item: item[2])
            beam = next_beam[:beam_width]
            
            # Optional: Add some randomization to beam selection to escape local minima
            # If we have more than beam_width, we already took the top ones.
            # To add diversity, we could randomly swap one of the top ones with a lower one.
            if len(next_beam) > beam_width and random.random() < 0.2:
                # Pick a random candidate from the rest
                rest = next_beam[beam_width:]
                if rest:
                    random_idx = random.randint(0, len(rest) - 1)
                    # Replace the last item in the beam (the worst of the top ones)
                    beam[-1] = rest[random_idx]

        # If we completed the row, return the best one (smallest max)
        # The beam contains full rows now
        # Sort by max entry
        beam.sort(key=lambda item: item[2])
        return beam[0][0]

    def greedy_row_fallback(used_other):
        """Standard greedy if beam search fails."""
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used_other) and not (new & row_diffs):
                row.append(x)
                row_diffs |= new
        return row

    # ---------- Per-row local optimization (refinement machinery) ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=20):
        """
        Try to find a row at row_idx with smaller max entry.
        Uses beam search with varying parameters.
        """
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            
            # Vary beam width and search depth/limit to explore different trade-offs
            # Early attempts: tight, small scope
            # Later attempts: wider beam, larger search window
            bw = 10 + (attempt % 5) * 5
            # Increase the search window limit for later attempts? 
            # The build_row_beam has a fixed limit logic. 
            # We can pass a 'search_limit' parameter if we modify build_row_beam.
            # For now, we rely on the internal logic.
            
            # To make it more diverse, we can randomly perturb the 'used_other' slightly?
            # No, used_other is fixed.
            
            # We can add randomization to the beam selection inside build_row_beam.
            # Let's just call it.
            new_row = build_row_beam(used_other, beam_width=bw)
            
            if new_row and max(new_row) < best_max:
                # Verify the whole set is valid with this new row
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = new_row
                if is_valid(test_rows):
                    best_row = list(new_row)
                    best_max = max(new_row)
                    
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 8
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=15)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """
        Perturb the solution to escape local minima.
        Uses beam search with higher randomness/larger beams to find diverse structures.
        """
        rows = [list(r) for r in rows]
        # Strategy: Rebuild 1 or 2 random rows
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Use a wider beam and more randomization for kicks
            # We can't easily change the internal randomization of build_row_beam without params.
            # So we'll just call it with a large beam width.
            # To ensure diversity, we might run it multiple times and pick a random valid one?
            # Or just pick the best.
            
            # Let's try a few different beam widths
            best_kick_row = None
            best_kick_max = max(rows[idx]) + 50 # Allow slightly larger rows for kicks
            
            for bw in [15, 25, 40]:
                if time_left() < 2:
                    break
                candidate_row = build_row_beam(used_other, beam_width=bw)
                if candidate_row:
                    # Check if it's valid in context
                    test_rows = [list(r) for r in rows]
                    test_rows[idx] = candidate_row
                    if is_valid(test_rows):
                        if max(candidate_row) < best_kick_max:
                            best_kick_max = max(candidate_row)
                            best_kick_row = candidate_row
            
            if best_kick_row:
                rows[idx] = best_kick_row
            else:
                # Fallback to greedy
                rows[idx] = greedy_row_fallback(used_other)

        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        elif time_left() > 30:
            # Additional refinement: try multiple kicks and pick the best
            best_kicked = polished_kicked
            for extra_kick in range(3):
                if time_left() < 15:
                    break
                kicked2 = kick(best_rows)
                polished2 = local_polish(kicked2)
                if scope(polished2) < scope(best_kicked):
                    best_kicked = polished2
            if scope(best_kicked) < best_scope:
                best_rows = best_kicked
                best_scope = scope(best_kicked)

    # Final high-precision polish pass
    if time_left() > 8:
        final_polish = local_polish(best_rows)
        if scope(final_polish) < best_scope:
            best_rows = final_polish
            best_scope = scope(final_polish)

    return best_rows
