import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # The problem: choose n_rows rows of (k+1) increasing integers starting at 0
    # such that all positive differences within rows are distinct across all rows.
    # Minimize the maximum entry (scope).

    # Total number of differences needed: n_rows * k*(k+1)/2
    # For n_rows=7, k=6: 7 * 21 = 147 differences
    # Counting bound: need at least 147 distinct differences, so max entry >= 147? No.
    # Actually the counting bound is based on: if max entry is M, then differences are in [1, M],
    # so we need M >= n_rows * k*(k+1)/2. For 7*21=147... wait that's not right either.
    # Let me re-read: "simple counting bound is 105". 
    # Hmm, for k=6, each row has C(7,2)=21 differences. 7 rows -> 147 differences.
    # But the counting bound says 105. Let me think again...
    # Actually wait - maybe the counting bound considers that differences must be in [1, M],
    # and we need 147 distinct values from [1, M], so M >= 147. But the prompt says 105.
    # 
    # Let me re-read: "Choose 7 rows of 6 increasing integers starting at 0"
    # So each row has 6 integers? Or 6+1=7 integers?
    # "k+1 integers that starts at 0 and increases strictly" - so k=6 means 7 integers per row.
    # Differences: C(7,2) = 21 per row. 7 rows -> 147 total differences.
    # Counting bound: M >= 147? But prompt says 105.
    #
    # Wait, maybe I'm misreading. Let me re-read more carefully.
    # "Choose 7 rows of 6 increasing integers starting at 0" - this says 6 integers.
    # But then "Each row is a list of k + 1 integers" - so if k=6, each row has 7 integers.
    # Hmm, there's a contradiction. Let me go with the interface: k+1 integers.
    # 
    # Actually wait - maybe "6 increasing integers" means 6 integers after the 0?
    # No, it says "starting at 0". So 6 integers total including 0?
    # But the interface says k+1 integers. If the task says k=6 and the row has 6 integers,
    # then k+1=6 means k=5? That doesn't match.
    #
    # Let me just trust the interface: construct(n_rows, k) returns rows of k+1 integers.
    # The evaluator passes n_rows=7 and k=6 (or whatever). Let me just handle it generically.
    #
    # Actually, re-reading: "Choose 7 rows of 6 increasing integers starting at 0"
    # I think "6" here might refer to k, and the row has k+1=7 integers. The "6 increasing"
    # might be a slight imprecision or means 6 steps. Let me just follow the interface.

    # Strategy: Use simulated annealing / local search to minimize the scope.
    # Start with a greedy construction, then try to reduce the scope by moving entries.

    def greedy_construct():
        """Greedy: build rows one by one, choosing smallest possible next entry."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if adding x creates any duplicate difference
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    for a in row:
                        used_diffs.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.append(r[j] - r[i])
        return diffs

    def is_valid(rows):
        """Check if all positive differences are distinct across all rows."""
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # Start with greedy
    rows = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(best_rows)
    
    # Also try a different greedy: build all rows simultaneously, round-robin
    def greedy_simultaneous():
        used_diffs = set()
        rows = [[0]] * n_rows
        # Each row needs k more entries
        remaining = [k] * n_rows
        x = 0
        while any(rem > 0 for rem in remaining):
            x += 1
            for i in range(n_rows):
                if remaining[i] <= 0:
                    continue
                row = rows[i]
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    for a in row:
                        used_diffs.add(x - a)
                    row.append(x)
                    remaining[i] -= 1
        return rows

    rows2 = greedy_simultaneous()
    if get_scope(rows2) < best_scope:
        best_rows = [r[:] for r in rows2]
        best_scope = get_scope(best_rows)
        rows = [r[:] for r in rows2]

    # Simulated annealing / local search
    # Move types:
    # 1. Change one entry in a row (keep it valid: increasing, starting at 0)
    # 2. Try to reduce the scope by lowering the maximum entry

    import copy

    def try_improve(rows, max_time):
        """Local search: try to reduce scope by adjusting entries."""
        current = [r[:] for r in rows]
        current_scope = get_scope(current)
        best_local = [r[:] for r in current]
        best_local_scope = current_scope
        t = time.time()
        
        # Collect all "hot" positions: entries that are at or near the max
        # Strategy: for each row, try to reduce the last entry, then second-to-last, etc.
        
        attempts = 0
        while time.time() < max_time:
            attempts += 1
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            # Pick a position other than 0 (which is fixed)
            pos = random.randint(1, k)  # positions 1..k (0-indexed, 0 is fixed at 0)
            
            old_val = current[row_idx][pos]
            # Try to decrease the value
            new_val = old_val - random.randint(1, max(1, old_val // 4))
            if new_val <= 0:
                new_val = 1
            
            # Ensure increasing order
            if pos > 0 and new_val <= current[row_idx][pos-1]:
                new_val = current[row_idx][pos-1] + 1
            if pos < k and new_val >= current[row_idx][pos+1]:
                new_val = current[row_idx][pos+1] - 1
                if new_val <= current[row_idx][pos-1] if pos > 0 else 0:
                    continue
            
            # Try the move
            current[row_idx][pos] = new_val
            
            # Check validity
            if is_valid(current):
                new_scope = get_scope(current)
                if new_scope < best_local_scope:
                    best_local_scope = new_scope
                    best_local = [r[:] for r in current]
            else:
                # Revert
                current[row_idx][pos] = old_val
        
        return best_local, best_local_scope

    # Run local search with time budget
    search_time = 120.0
    while time.time() < start_time + search_time:
        rows, scope = try_improve(rows, time.time() + min(10.0, search_time - (time.time() - start_time)))
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            rows = [r[:] for r in best_rows]
        else:
            # Keep current rows for next iteration
            rows = [r[:] for r in rows]
        
        if time.time() > deadline - 15.0:
            break

    # Final polish: try to reduce scope further with targeted moves
    if time.time() < deadline - 5.0:
        # Try reducing the maximum entry directly
        max_val = get_scope(best_rows)
        for row_idx in range(n_rows):
            for pos in range(k, 0, -1):
                if best_rows[row_idx][pos] == max_val:
                    # Try to reduce this entry
                    for delta in range(1, min(20, max_val)):
                        new_val = max_val - delta
                        # Check if it maintains increasing order
                        if pos > 0 and new_val <= best_rows[row_idx][pos-1]:
                            continue
                        if pos < k and new_val >= best_rows[row_idx][pos+1]:
                            continue
                        # Try it
                        old = best_rows[row_idx][pos]
                        best_rows[row_idx][pos] = new_val
                        if is_valid(best_rows):
                            max_val = get_scope(best_rows)
                            break
                        else:
                            best_rows[row_idx][pos] = old
    
    return best_rows
# EVOLVE-BLOCK-END
