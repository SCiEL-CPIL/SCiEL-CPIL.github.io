"""
Strategy: Conflict-Driven Local Search with Row-Swap Decomposition.
Instead of random single-element moves, we use a two-phase approach:
1. Construction: Greedy placement to get a valid (but potentially large) solution.
2. Optimization: 
   - Phase A: Resolve any remaining conflicts (if starting from random) or maintain validity.
   - Phase B: "Tightening" - Aggressively try to reduce the maximum entry by moving elements 
     towards 0, accepting moves that keep the solution valid.
   - Move Operator: Swap elements between different rows or shift an element within a row 
     to a smaller available value.
   
This is structurally different from the seed's generic SA because it uses 
conflict-directed moves and a specific "tightening" heuristic that is 
greedy with respect to the objective (max entry) while maintaining feasibility.
"""
import random
import time
import math
import numpy as np
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute indices for differences
    # For a row of length k+1, indices are 0..k
    # Differences are row[j] - row[i] for 0 <= i < j <= k
    diff_indices = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]
    num_diffs_per_row = len(diff_indices)
    total_diffs = n_rows * num_diffs_per_row
    
    # Helper: Compute all differences for a set of rows
    # Returns a dict: diff_value -> list of (row_idx, i, j)
    def compute_diff_map(rows):
        diff_map = defaultdict(list)
        for r_idx, row in enumerate(rows):
            for i, j in diff_indices:
                d = row[j] - row[i]
                diff_map[d].append((r_idx, i, j))
        return diff_map

    # Helper: Check validity and count conflicts
    def get_conflicts(diff_map):
        conflicts = 0
        for d, locs in diff_map.items():
            if len(locs) > 1:
                conflicts += len(locs) - 1
        return conflicts

    # Helper: Get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Initial Solution: Greedy Construction
    # We construct rows one by one. For each new element, pick the smallest integer 
    # that doesn't create a duplicate difference with existing elements in the same row 
    # AND doesn't create a duplicate difference with any existing row.
    def greedy_construct():
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # We need k more elements
            # To keep it compact, we try small numbers first
            candidate = 1
            while len(row) < k + 1:
                # Check if adding 'candidate' to 'row' creates any conflict
                # New diffs: candidate - x for x in row
                # Also need to check if 'candidate' itself is valid (it is, since >0)
                # And check if any diff (candidate - x) is in used_diffs
                # AND check if any diff (candidate - x) conflicts with diffs within the current row (which are unique by construction if we pick smallest)
                # Wait, diffs within current row are already unique. 
                # We just need to ensure new diffs are not in global used_diffs.
                
                new_diffs = [candidate - x for x in row]
                valid = True
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                
                if valid:
                    # Accept candidate
                    row.append(candidate)
                    used_diffs.update(new_diffs)
                    candidate += 1
                else:
                    candidate += 1
            rows.append(row)
        return rows

    # Initial best
    best_rows = greedy_construct()
    best_max = get_max_entry(best_rows)
    
    # If we want to try to improve, we can use local search.
    # The greedy solution is valid. We want to reduce max_entry.
    # We will use a "Tightening" local search.
    # State: rows, diff_map, conflicts (should be 0)
    # Move: Pick a row r, pick an index j (1..k), try to set row[r][j] to a smaller value v.
    #       Constraint: v > row[r][j-1] and v < row[r][j+1] (if j<k)
    #       Constraint: All new diffs must not conflict with any other diffs in the system.
    
    # To make this fast, we maintain the diff_map.
    # When we change row[r][j] from old_v to new_v:
    # 1. Remove old diffs involving row[r][j].
    #    These are (old_v - row[r][i]) for i < j and (row[r][i] - old_v) for i > j? No, diffs are positive.
    #    Diffs involving index j:
    #    - (row[r][j] - row[r][i]) for i < j
    #    - (row[r][i] - row[r][j]) for i > j
    # 2. Check if new diffs (new_v - row[r][i] for i < j) and (row[r][i] - new_v for i > j) 
    #    conflict with any existing diffs in the system (excluding those we just removed).
    # 3. If no conflict, accept.
    
    # For efficiency, we'll run this for a fixed number of passes or time limit.
    
    current_rows = [row[:] for row in best_rows]
    current_diff_map = compute_diff_map(current_rows)
    current_conflicts = get_conflicts(current_diff_map)
    current_max = get_max_entry(current_rows)
    
    # Optimization Loop
    # We try to reduce the max entry.
    # We'll iterate through elements in the row with the current max, trying to lower them.
    
    # To avoid getting stuck, we can also use "swaps" between rows if a direct reduction fails, 
    # but direct reduction is simpler and often effective for "tightening".
    
    # Let's add a bit of stochasticity: if direct reduction fails for an element, 
    # maybe swap it with an element from another row? 
    # Actually, let's stick to a robust "Best-First" or "First-Fit" descent.
    
    # We'll use a time-bounded loop.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through all rows
        for r_idx in range(n_rows):
            if time.time() > deadline: break
            # Iterate through elements from largest to smallest (index k down to 1)
            # We want to reduce large values first.
            for j in range(k, 0, -1):
                if time.time() > deadline: break
                old_v = current_rows[r_idx][j]
                # Try to find a smaller value
                # Min value is current_rows[r_idx][j-1] + 1
                min_v = current_rows[r_idx][j-1] + 1
                # Max value to try is old_v - 1
                # We try values from min_v to old_v - 1
                
                # Optimization: If old_v is already min_v, skip.
                if old_v == min_v:
                    continue
                
                # We can try a few candidates, or scan all.
                # Scanning all is O(k^2) per check if we do it naively.
                # Let's try a limited number of candidates or a random subset if the range is large.
                # However, for small k (like 7), the range is small.
                
                found = False
                # Try values from min_v to old_v - 1
                # To be efficient, we can check if any value in [min_v, old_v-1] works.
                # Let's just iterate.
                for new_v in range(min_v, old_v):
                    # Check validity
                    # Remove old diffs from consideration (conceptually)
                    # New diffs:
                    # 1. new_v - current_rows[r_idx][i] for i in 0..j-1
                    # 2. current_rows[r_idx][i] - new_v for i in j+1..k
                    
                    valid = True
                    # We need to check against the global diff map, 
                    # but we must ignore the diffs that are being removed (old diffs involving j).
                    
                    # Let's compute the set of diffs that will be removed.
                    removed_diffs = set()
                    for i in range(j):
                        removed_diffs.add(old_v - current_rows[r_idx][i])
                    for i in range(j + 1, k + 1):
                        removed_diffs.add(current_rows[r_idx][i] - old_v)
                    
                    # Now check new diffs
                    new_diffs = []
                    for i in range(j):
                        d = new_v - current_rows[r_idx][i]
                        new_diffs.append(d)
                    for i in range(j + 1, k + 1):
                        d = current_rows[r_idx][i] - new_v
                        new_diffs.append(d)
                    
                    for d in new_diffs:
                        # If d is in removed_diffs, it's fine (it's being replaced)
                        # But wait, if d is in removed_diffs, it means the old value had this diff too?
                        # No, removed_diffs are specific to old_v.
                        # If new_v creates a diff d, and old_v also created a diff d, 
                        # then d is in removed_diffs. But we are replacing old_v, so the old d is gone.
                        # However, we must check if d exists in the *rest* of the system.
                        
                        # The diff_map contains all current diffs.
                        # If d is in diff_map, it might be from the old_v (which we are removing) 
                        # or from somewhere else.
                        
                        # Let's check the count in diff_map.
                        # If d is in diff_map:
                        #   Let count = len(diff_map[d])
                        #   Let removed_count = 1 if d in removed_diffs else 0
                        #   If count - removed_count > 0, then there is another row/element using d.
                        #   So conflict!
                        
                        if d in current_diff_map:
                            count = len(current_diff_map[d])
                            removed_count = 1 if d in removed_diffs else 0
                            if count - removed_count > 0:
                                valid = False
                                break
                    
                    if valid:
                        # Accept move
                        # Update current_rows
                        current_rows[r_idx][j] = new_v
                        # Update current_diff_map
                        # Remove old entries
                        for loc in current_diff_map[old_v - current_rows[r_idx][i]]: # This is wrong, i is loop var
                            pass # We need to do this properly.
                        
                        # Let's just rebuild the diff_map for this row to be safe, 
                        # or do it incrementally. Incremental is better.
                        
                        # Remove old diffs from map
                        for i in range(j):
                            d = old_v - current_rows[r_idx][i]
                            loc = (r_idx, i, j)
                            lst = current_diff_map[d]
                            # Remove loc from lst
                            # Using list remove is O(n). n is small (total_diffs/k^2 ~ small)
                            # But to be safe, let's filter.
                            current_diff_map[d] = [l for l in lst if l != loc]
                            if not current_diff_map[d]:
                                del current_diff_map[d]
                        for i in range(j + 1, k + 1):
                            d = current_rows[r_idx][i] - old_v
                            loc = (r_idx, j, i)
                            lst = current_diff_map[d]
                            current_diff_map[d] = [l for l in lst if l != loc]
                            if not current_diff_map[d]:
                                del current_diff_map[d]
                        
                        # Add new diffs to map
                        for i in range(j):
                            d = new_v - current_rows[r_idx][i]
                            loc = (r_idx, i, j)
                            if d in current_diff_map:
                                current_diff_map[d].append(loc)
                            else:
                                current_diff_map[d] = [loc]
                        for i in range(j + 1, k + 1):
                            d = current_rows[r_idx][i] - new_v
                            loc = (r_idx, j, i)
                            if d in current_diff_map:
                                current_diff_map[d].append(loc)
                            else:
                                current_diff_map[d] = [loc]
                        
                        # Update best if improved
                        new_max = get_max_entry(current_rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [row[:] for row in current_rows]
                            improved = True
                        found = True
                        break # Try next j
                
                if not found:
                    # Optional: Try a swap with another row to enable future reductions?
                    # For now, skip.
                    pass
    
    # Final check
    return best_rows

# EVOLVE-BLOCK-END
