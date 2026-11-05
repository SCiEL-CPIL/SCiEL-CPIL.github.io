"""
Difference triangle set (7,5) solver.

Strategy:
1. Enumerate all valid "frames" (rows of k+1 increasing integers starting at 0)
   up to a maximum scope M.
2. Use a backtracking search over the set of frames, selecting the next frame
   using a "minimum remaining values" heuristic based on a pre-computed
   compatibility graph (bitset-based for speed).
3. Use a fast greedy column-wise construction for an initial solution.
4. Polish with local search.
"""
import time
import random
import sys
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    all_diffs = []
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.append(r[j] - r[i])
    return len(all_diffs) == len(set(all_diffs))

def _compute_all_diffs(rows):
    """Return set of all differences used."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    return used

def _enumerate_frames(M, k):
    """
    Enumerate all valid frames (rows) up to max value M.
    A frame is a tuple of k+1 increasing integers starting at 0.
    Returns a list of (frame_tuple, diff_mask_int) where diff_mask_int
    is a bitmask of the differences used.
    """
    frames = []
    # Generate all combinations of k values from 1..M
    # The row is (0, a1, a2, ..., ak)
    for combo in combinations(range(1, M + 1), k):
        row = (0,) + combo
        # Check if all differences within the row are distinct
        diffs = set()
        valid = True
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
        if valid:
            # Create bitmask
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            frames.append((row, mask))
    return frames

def _greedy_construct(n_rows, k, deadline):
    """
    Fast greedy construction.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, k + 1):
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        for idx in row_order:
            row = rows[idx]
            prev_val = row[-1]
            x = prev_val + 1
            while True:
                if time.time() > deadline:
                    return rows
                new_diffs = set()
                conflict = False
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    row.append(x)
                    used_diffs |= new_diffs
                    break
                x += 1
    return rows

def _local_search(rows, k, deadline):
    """
    Local search to reduce scope.
    """
    rows = [r[:] for r in rows]
    used = _compute_all_diffs(rows)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    # Check if move is valid
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                            new_diffs.add(new_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                            new_diffs.add(row[i] - new_val)
                    
                    remaining = used - old_diffs
                    if new_diffs & remaining:
                        continue
                    
                    # Move is valid
                    row[pos] = new_val
                    used = (used - old_diffs) | new_diffs
                    improved = True
                    break
    return rows

def _backtrack_csp(frames, n_rows, k, deadline):
    """
    Backtracking search over pre-computed frames.
    frames: list of (row_tuple, mask_int)
    """
    if not frames:
        return None
    
    # Sort frames by max value to try smaller scopes first
    frames.sort(key=lambda x: max(x[0]))
    
    # Pre-compute compatibility: for each frame, which other frames are compatible?
    # Since n_rows is small (7), we can just check on the fly with bitsets.
    
    best_solution = None
    
    def search(current_rows, used_mask, start_idx):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(current_rows) == n_rows:
            best_solution = [list(r) for r in current_rows]
            return
        
        # Heuristic: try frames that are "most constrained" or just iterate
        # For simplicity, iterate from start_idx to avoid permutations
        # We want to find ANY solution, so order matters for speed.
        
        for i in range(start_idx, len(frames)):
            row, mask = frames[i]
            
            # Check compatibility
            if used_mask & mask:
                continue
            
            current_rows.append(row)
            search(current_rows, used_mask | mask, i + 1)
            
            if best_solution is not None:
                return
            
            current_rows.pop()
    
    search([], 0, 0)
    return best_solution

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Step 1: Fast greedy initial solution
    rows = _greedy_construct(n_rows, k, deadline - 10)
    if rows is None or len(rows) != n_rows:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
    
    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    
    # Step 2: Local search polish
    if time.time() < deadline - 5:
        polished = _local_search(best_rows, k, deadline - 5)
        sc = _scope(polished)
        if sc < best_scope:
            best_scope = sc
            best_rows = polished
    
    # Step 3: CSP Search
    # We want to find a solution with scope < best_scope.
    # Let's try to find the minimum possible scope.
    # Counting bound: n_rows * k * (k+1) / 2
    # For 7 rows, k=5: 7 * 15 / 2 = 52.5 -> 53.
    # But differences must be distinct. Total diffs = 7 * 15 = 105.
    # So max diff must be at least 105.
    
    lower_bound = n_rows * k * (k + 1) // 2 + 1
    if lower_bound < 105:
        lower_bound = 105 # Heuristic lower bound based on total diffs
    
    # Try to find a solution with scope = lower_bound, lower_bound+1, etc.
    # But enumerating all frames up to M=105 might be slow or memory heavy?
    # Number of frames up to M=105 with k=5: C(105, 5) is large (~100 million).
    # We cannot enumerate all frames up to 105.
    
    # Instead, let's use the CSP search on a restricted set of frames.
    # Or, let's stick to the local search and maybe a smarter backtracking on the values directly.
    
    # Let's try a direct backtracking on the values for the first few rows to see if we can beat the greedy.
    # This is similar to the seed program's _backtrack_construct but we can optimize it.
    
    # Given the time limit, let's try a few random restarts of a smarter greedy.
    rng = random.Random(42)
    
    for attempt in range(10):
        if time.time() > deadline - 2:
            break
        
        # Randomized greedy
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        success = True
        for pos in range(1, k + 1):
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            for idx in row_order:
                row = rows[idx]
                prev_val = row[-1]
                x = prev_val + 1
                found = False
                
                # Try a window of values
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    new_diffs = set()
                    conflict = False
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if not conflict:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                    x += 1
                
                if not found:
                    success = False
                    break
            
            if not success:
                break
        
        if success and len(rows[0]) == k + 1:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]
                
                # Polish the new best
                if time.time() < deadline - 2:
                    polished = _local_search(best_rows, k, deadline - 2)
                    sc2 = _scope(polished)
                    if sc2 < best_scope:
                        best_scope = sc2
                        best_rows = polished
    
    return best_rows

# EVOLVE-BLOCK-END
