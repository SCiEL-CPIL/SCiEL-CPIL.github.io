import time
import sys

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _solve_for_m(n_rows, k, m):
    """
    Try to find n_rows rows of k+1 marks (starting at 0) with max entry <= m.
    Uses backtracking with forward checking and symmetry breaking (lexicographic rows).
    """
    # k=5 means 6 marks per row.
    # Total differences per row: k*(k+1)/2 = 15.
    # Total differences: n_rows * 15.
    
    # We will represent used differences as a set for simplicity, 
    # but for speed, a boolean array of size m+1 is better.
    
    # Symmetry breaking: rows must be in lexicographic order.
    # Row 0 is the "smallest".
    
    rows = []
    used_diffs = [False] * (m + 1)
    
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_lex_greater_or_equal(row, prev_row):
        if prev_row is None:
            return True
        if len(row) != len(prev_row):
            return len(row) > len(prev_row)
        for i in range(len(row)):
            if row[i] < prev_row[i]:
                return False
            if row[i] > prev_row[i]:
                return True
        return True

    def backtrack(row_idx, current_row, prev_row):
        if row_idx == n_rows:
            return True
        
        if len(current_row) == k + 1:
            # Row is complete
            # Check lexicographic constraint
            if not is_lex_greater_or_equal(current_row, prev_row):
                return False
            
            # Check differences
            diffs = get_row_diffs(current_row)
            # Check for duplicates within the row (should be none if we built it correctly, but just in case)
            if len(set(diffs)) != len(diffs):
                return False
            
            # Check against global used_diffs
            for d in diffs:
                if d > m:
                    return False
                if used_diffs[d]:
                    return False
            
            # Add to used_diffs
            for d in diffs:
                used_diffs[d] = True
            
            rows.append(current_row)
            
            if backtrack(row_idx + 1, [0], current_row):
                return True
            
            # Backtrack
            rows.pop()
            for d in diffs:
                used_diffs[d] = False
            return False
        
        # Determine next value
        if len(current_row) == 0:
            candidates = [0]
        else:
            last_val = current_row[-1]
            # We must pick a value > last_val
            # Also, we need to ensure we can fit the remaining marks.
            # Remaining marks: (k+1) - len(current_row)
            # We need to pick these many values from last_val+1 to m.
            # So last_val + remaining <= m is not strictly required, but 
            # last_val + 1 + (remaining - 1) <= m => last_val + remaining - 1 <= m.
            # So last_val <= m - remaining + 1.
            remaining = (k + 1) - len(current_row)
            max_next = m - remaining + 1
            if max_next < last_val + 1:
                return False
            candidates = range(last_val + 1, max_next + 1)
        
        for v in candidates:
            # Check differences with existing marks in current_row
            valid = True
            new_diffs = []
            for a in current_row:
                d = v - a
                if d > m:
                    valid = False
                    break
                if used_diffs[d]:
                    valid = False
                    break
                # Also check if this difference is already in the current row's partial diffs?
                # No, we haven't added them to global yet. But we must ensure they are distinct 
                # from each other and from the global set.
                # Since we check `used_diffs`, and we haven't added the current row's diffs to `used_diffs` yet,
                # we need to ensure the new diff `d` is not equal to any other new diff in this step?
                # Actually, `v-a` for different `a` are distinct because `a` are distinct.
                # But we need to ensure `d` is not already used in a previous row.
                # That's what `used_diffs` does.
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # If valid, tentatively add
            current_row.append(v)
            
            # Note: We do NOT add to used_diffs yet, because the row is not complete.
            # We only add to used_diffs when the row is complete.
            # This is a problem: we might pick a value that creates a diff that is 
            # already used by another part of the SAME row? No, diffs within a row are 
            # generated by pairs. If we pick v, we generate diffs v-a for all a in row.
            # Are these diffs distinct from each other? Yes, because a is distinct.
            # Are they distinct from diffs generated by previous steps in this row?
            # Yes, because those were generated by (v'-a') where v' < v.
            # Wait, if we pick v1 then v2, we have diffs v1-a, v2-a, v2-v1.
            # If we pick v3, we have diffs v3-a, v3-v1, v3-v2.
            # It is possible that v3-v1 == v2-a?
            # Yes. So we MUST check new diffs against the diffs already present in the current row.
            
            # Let's re-evaluate.
            # We need to maintain a set of diffs for the current row.
            # Let's pass `current_row_diffs` as a set or list.
            
            # This changes the signature of backtrack.
            # Let's refactor.
            
            # Actually, we can just check if any new diff is in the set of diffs 
            # already formed by the current row prefix.
            
            # Let's compute the set of diffs in the current prefix.
            # This is inefficient if done repeatedly.
            # Better to maintain it.
            
            # Let's restart the backtrack function definition with a `row_diffs_set` parameter.
            # But for now, let's just check against the global `used_diffs` 
            # and assume we handle row-internal conflicts by checking against 
            # the diffs of the current row prefix.
            
            # Let's optimize: 
            # When we add v, we generate diffs D_new = {v - a for a in current_row}.
            # We must ensure D_new is disjoint from `used_diffs` (global) 
            # AND disjoint from `current_row_diffs` (local).
            
            # Let's implement a helper to get current row diffs.
            # For k=5, row length is 6. Max 15 diffs.
            # Checking 15 diffs against a set of 15 diffs is fast.
            
            # Let's try this approach in the loop.
            
            # Compute current row diffs
            curr_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    curr_diffs.add(current_row[j] - current_row[i])
            
            conflict = False
            for d in new_diffs:
                if d in used_diffs or d in curr_diffs:
                    conflict = True
                    break
            
            if conflict:
                current_row.pop()
                continue
            
            # Recurse
            if backtrack(row_idx, current_row, prev_row):
                return True
            
            # Backtrack
            current_row.pop()
        
        return False

    # We need to update the backtrack call to pass row_diffs or compute it.
    # The above code computes curr_diffs inside the loop, which is O(k^2) per candidate.
    # This is acceptable for small k.
    
    # However, the `used_diffs` update happens only when the row is complete.
    # This means that while building a row, we are not blocking diffs 
    # that are used by the SAME row in future steps?
    # No, we check `curr_diffs` which contains all diffs from the prefix.
    # So if a future diff conflicts with a past diff in the same row, it will be caught.
    
    # One issue: `used_diffs` is global. 
    # When we backtrack from a complete row, we remove its diffs from `used_diffs`.
    # This is correct.
    
    # Let's run the solver.
    if backtrack(0, [0], None):
        return rows
    return None

def construct(n_rows, k):
    deadline = time.time() + 140  # 140s safety margin
    
    # Lower bound
    # Total diffs = n_rows * k*(k+1)/2
    # Min M such that M >= Total Diffs
    total_diffs = n_rows * (k * (k + 1) // 2)
    min_m = total_diffs
    
    # Known best is 112. 
    # We try from min_m up to some limit.
    # If we find a solution for M, we return it.
    # Since we want to minimize M, we stop at the first M that works.
    
    # To be safe, we might want to try a few random restarts or different orderings 
    # if the first M fails quickly. But backtracking with lex ordering is deterministic.
    # If it fails, it fails.
    
    # However, the search space can be huge.
    # We might need to add a timeout inside the backtracking to switch to a heuristic 
    # or try a slightly larger M if the current one is too hard.
    
    # For now, let's try M from min_m to 112 (or a bit higher).
    
    for m in range(min_m, 115):
        if time.time() > deadline:
            break
        
        # Try to solve for this M
        # To avoid hanging on a hard instance, we can add a node limit 
        # or a time limit to the backtracking.
        # But for small M (105-112), it should be fast if a solution exists.
        # If no solution exists, it will explore the whole tree, which is bad.
        
        # Let's add a simple node counter limit to backtrack.
        # If it exceeds, we abort this M and try the next.
        
        node_count = 0
        max_nodes = 1000000  # 1 million nodes
        
        # We need to modify backtrack to respect this.
        # Let's wrap it.
        
        def backtrack_limited(row_idx, current_row, prev_row):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                if not is_lex_greater_or_equal(current_row, prev_row):
                    return False
                
                diffs = []
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        diffs.append(current_row[j] - current_row[i])
                
                if len(set(diffs)) != len(diffs):
                    return False
                
                for d in diffs:
                    if d > m:
                        return False
                    if used_diffs[d]:
                        return False
                
                for d in diffs:
                    used_diffs[d] = True
                
                rows.append(current_row)
                
                if backtrack_limited(row_idx + 1, [0], current_row):
                    return True
                
                rows.pop()
                for d in diffs:
                    used_diffs[d] = False
                return False
            
            if len(current_row) == 0:
                candidates = [0]
            else:
                last_val = current_row[-1]
                remaining = (k + 1) - len(current_row)
                max_next = m - remaining + 1
                if max_next < last_val + 1:
                    return False
                candidates = range(last_val + 1, max_next + 1)
            
            for v in candidates:
                valid = True
                new_diffs = []
                for a in current_row:
                    d = v - a
                    if d > m:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check against current row diffs
                curr_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        curr_diffs.add(current_row[j] - current_row[i])
                
                conflict = False
                for d in new_diffs:
                    if d in curr_diffs:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                current_row.append(v)
                
                if backtrack_limited(row_idx, current_row, prev_row):
                    return True
                
                current_row.pop()
            
            return False

        rows = []
        used_diffs = [False] * (m + 1)
        
        # Note: is_lex_greater_or_equal and get_row_diffs are defined in _solve_for_m, 
        # but we are re-implementing the logic here to avoid scope issues.
        # Let's move the helper functions to module level or pass them.
        
        # For simplicity, I'll just define them inside construct.
        
        def is_lex_greater_or_equal(row, prev_row):
            if prev_row is None:
                return True
            if len(row) != len(prev_row):
                return len(row) > len(prev_row)
            for i in range(len(row)):
                if row[i] < prev_row[i]:
                    return False
                if row[i] > prev_row[i]:
                    return True
            return True

        # Run the limited backtracker
        if backtrack_limited(0, [0], None):
            # Verify the solution
            if _validate(rows):
                return rows
    
    # If we didn't find a solution within the time or M range, 
    # fall back to a greedy solution to ensure we return something valid.
    # The greedy solution might have a larger M, but it's valid.
    
    def _greedy_row(used, k):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row

    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    
    return rows
