import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # Greedy baseline
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

    # Backtracking with beam search
    def beam_search(max_beam=200, max_depth_limit=200):
        """Build rows using beam search with forward checking."""
        # State: (rows_so_far, used_diffs_set, current_scope)
        # We build row by row, position by position
        
        # Start with one partial row: [0]
        beam = [([0], set({}))]
        
        for row_idx in range(n_rows):
            if time_left() < 5:
                break
            new_beam = []
            for partial_row, used_diffs in beam:
                if time_left() < 5:
                    break
                # Try to extend this row
                row = list(partial_row)
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])
                
                # If row is complete, move to next row
                if len(row) == k + 1:
                    new_beam.append((row, used_diffs | row_diffs))
                    continue
                
                # Find next position to fill (just append to end)
                last = row[-1]
                # Try candidate values
                candidates = []
                for x in range(last + 1, last + 50):  # Search window
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs) and not (new_diffs & row_diffs):
                        candidates.append((x, len(new_diffs)))
                
                # Sort by value (prefer smaller)
                candidates.sort()
                
                # Add top candidates to new beam
                for x, _ in candidates[:3]:
                    new_row = row + [x]
                    new_row_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            new_row_diffs.add(new_row[j] - new_row[i])
                    new_beam.append((new_row, used_diffs | new_row_diffs))
            
            if not new_beam:
                break
            
            # Keep only top beam_size states
            # Score by: (max_entry, sum_of_entries) - prefer tighter packing
            new_beam.sort(key=lambda s: (max(s[0]), sum(s[0])))
            beam = new_beam[:max_beam]
        
        # Convert beam to complete solutions
        results = []
        for partial_row, used_diffs in beam:
            if len(partial_row) == k + 1:
                results.append(partial_row)
        
        return results

    # Local search: try to reduce scope by modifying individual entries
    def local_search(rows, max_iters=500):
        rows = [list(r) for r in rows]
        improved = True
        iters = 0
        while improved and iters < max_iters and time_left() > 10:
            improved = False
            iters += 1
            for row_idx in range(n_rows):
                if time_left() < 10:
                    break
                row = rows[row_idx]
                used_other = set()
                for j in range(n_rows):
                    if j != row_idx:
                        for i in range(len(rows[j])):
                            for l in range(i + 1, len(rows[j])):
                                used_other.add(rows[j][l] - rows[j][i])
                
                # Try to reduce each element (except first which is 0)
                for pos in range(len(row) - 1, 0, -1):  # From end to start
                    old_val = row[pos]
                    # Try smaller values
                    for new_val in range(row[pos-1] + 1, old_val):
                        test_row = row[:pos] + [new_val] + row[pos+1:]
                        # Check if test_row is increasing
                        if any(test_row[i] >= test_row[i+1] for i in range(len(test_row)-1)):
                            continue
                        # Check differences
                        test_diffs = set()
                        for i in range(len(test_row)):
                            for j in range(i + 1, len(test_row)):
                                d = test_row[j] - test_row[i]
                                if d in used_other or d in test_diffs:
                                    break
                                test_diffs.add(d)
                        else:
                            # Valid
                            rows[row_idx] = test_row
                            improved = True
                            break
                    if improved:
                        break
        return rows

    # Run beam search with different beam widths
    for beam_width in [100, 200, 400]:
        if time_left() < 15:
            break
        results = beam_search(max_beam=beam_width)
        for res in results:
            if is_valid([res] + best_rows[1:]):
                candidate = [res] + best_rows[1:]
                if is_valid(candidate) and scope(candidate) < best_scope:
                    best_rows = candidate
                    best_scope = best_scope if False else scope(candidate)
        # Also try with other row positions
        for pos in range(1, n_rows):
            if time_left() < 10:
                break
            for res in results:
                candidate = best_rows[:pos] + [res] + best_rows[pos+1:]
                if is_valid(candidate) and scope(candidate) < best_scope:
                    best_rows = candidate
                    best_scope = scope(candidate)

    # Local search to refine
    if time_left() > 15:
        polished = local_search(best_rows)
        if is_valid(polished) and scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # ILS: kick + local search
    def kick(rows):
        rows = [list(r) for r in rows]
        idx = random.randint(0, n_rows - 1)
        row = rows[idx]
        used_other = set()
        for j in range(n_rows):
            if j != idx:
                for i in range(len(rows[j])):
                    for l in range(i + 1, len(rows[j])):
                        used_other.add(rows[j][l] - rows[j][i])
        
        # Rebuild row with randomness
        new_row = [0]
        row_diffs = set()
        for _ in range(k):
            last = new_row[-1]
            found = False
            for x in range(last + 1, last + 100):
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                    # Sometimes skip to add randomness
                    if random.random() < 0.1 and x < 300:
                        continue
                    new_row.append(x)
                    row_diffs |= new_diffs
                    found = True
                    break
            if not found:
                break
        if len(new_row) == k + 1:
            rows[idx] = new_row
        return rows

    consecutive_no_improve = 0
    max_no_improve = 30
    while time_left() > 20 and consecutive_no_improve < max_no_improve:
        kicked = kick(best_rows)
        if not is_valid(kicked):
            consecutive_no_improve += 1
            continue
        
        polished = local_search(kicked, max_iters=200)
        if is_valid(polished) and scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return best_rows
