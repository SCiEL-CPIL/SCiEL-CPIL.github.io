import time
import random
import sys
import numpy as np

# ---------------- global scratch buffers ----------------
_MAXK = 64
_MAXROWS = 64
# We will use Python sets for used differences in the local search for simplicity and correctness,
# or a numpy bitset if we can manage it cleanly.
# Given the previous failure with numpy bitsets, let's use a Python set for `used_diffs` 
# inside the local search, as the number of differences is small (4 * 7*8/2 = 112 diffs).
# A set of 112 integers is very fast for membership checks.

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 117.0

    k1 = k + 1
    # Lower bound on scope: need n_rows * k*(k+1)/2 distinct positive differences
    # Actually, the bound is roughly sqrt(2 * total_diffs) if they were packed tightly, 
    # but here they are spread. 
    # Simple bound: max_entry >= sqrt(2 * n_rows * k*(k+1)/2) ? No.
    # The counting bound mentioned is 112 for (4,7).
    # total_diffs = 4 * 28 = 112.
    # If scope is M, max diffs is M. So M >= 112.
    
    best_solution = None
    best_scope = float('inf')
    
    # 1) Initial Greedy Solution
    best_solution = _greedy_random(n_rows, k, seed=0)
    if best_solution:
        best_scope = max(max(r) for r in best_solution)
    
    # 2) Local Search Loop
    # We will try to improve the solution by "pulling" elements to the left.
    # We will also try random restarts if we get stuck.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 2.0:
        # Run local search on the current best
        current_sol = [list(r) for r in best_solution]
        current_scope = max(max(r) for r in current_sol)
        
        # Local Search: Iterate through rows and positions, try to minimize values
        # We do multiple passes until no improvement
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for ri in range(n_rows):
                r = current_sol[ri]
                # Process positions from right to left? Or left to right?
                # Left to right allows smaller values for earlier elements, which might constrain later ones.
                # Right to left allows pulling later elements in, which might free up space for earlier ones?
                # Let's try Right to Left (k down to 1) to minimize the max entry directly.
                for pos in range(k, 0, -1):
                    cur_val = r[pos]
                    # Find smallest valid value > r[pos-1]
                    # We need to check conflicts with:
                    # 1. Other elements in the same row
                    # 2. Elements in other rows (via global used set)
                    
                    # Build the set of "forbidden" differences for this position
                    # Forbidden diffs are:
                    # a) Diffs used by OTHER rows
                    # b) Diffs used by OTHER positions in THIS row (excluding the current one's old diffs)
                    
                    # Efficiently: 
                    # 1. Compute all diffs in the current solution.
                    # 2. Remove diffs contributed by r[pos].
                    # 3. Try new values.
                    
                    # To avoid recalculating all diffs every time, we can maintain a global used_set.
                    # But updating it during the search is tricky.
                    # Given small size (112 diffs), recalculating is O(112 * 112) ~ 12k ops.
                    # Doing this for 36 positions * 4 rows * 100 iterations = 1.4M ops. Fast.
                    
                    # Let's just rebuild the used set for the *entire* solution minus the current element's contribution?
                    # Or simpler: Check validity of candidate `nv` against the *current* full set, 
                    # but ignore the conflicts with `r[pos]` itself?
                    # No, if we change `r[pos]` to `nv`, the old diffs of `r[pos]` are removed, 
                    # and new diffs of `nv` are added.
                    # So `nv` is valid if:
                    # (New diffs of nv) are not in (Old Used Set - Old Diffs of r[pos]).
                    
                    old_used = _get_used_diffs(current_sol)
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(cur_val - r[i])
                    for j in range(pos + 1, k1):
                        old_diffs.add(r[j] - cur_val)
                    
                    available = old_used - old_diffs
                    
                    # Try values from r[pos-1]+1 to cur_val
                    found = False
                    for nv in range(r[pos - 1] + 1, cur_val + 1):
                        # Check if nv creates any diff in `available`
                        conflict = False
                        # Diffs with earlier
                        for i in range(pos):
                            d = nv - r[i]
                            if d in available:
                                conflict = True
                                break
                        if conflict:
                            continue
                        # Diffs with later
                        for j in range(pos + 1, k1):
                            d = r[j] - nv
                            if d in available:
                                conflict = True
                                break
                        if conflict:
                            continue
                        
                        # Valid!
                        r[pos] = nv
                        improved = True
                        found = True
                        break
                    
                    if found:
                        # Since r changed, the loop for this row can continue to next pos
                        # But we should probably restart the row or just continue?
                        # Continuing is fine, as we are moving left.
                        pass

        # After local search, check if we improved
        new_scope = max(max(r) for r in current_sol)
        if new_scope < best_scope:
            best_scope = new_scope
            best_solution = current_sol
            # If we hit the theoretical lower bound, we can stop
            if best_scope <= 112:
                break
        else:
            # Stagnation: Try a random restart or perturbation
            # Generate a new random solution
            seed = int(time.time() * 1000) % 100000
            new_sol = _greedy_random(n_rows, k, seed=seed)
            if new_sol:
                # Polish it
                # We can run the same local search logic here, or just accept if it's better
                # For now, just accept if better
                s = max(max(r) for r in new_sol)
                if s < best_scope:
                    best_solution = new_sol
                    best_scope = s
            # Also try perturbing the current best
            # Pick a random row and random position, increase its value by a random amount, then local search?
            # This is complex. Let's stick to random restarts for now.

    return best_solution

def _get_used_diffs(rows):
    used = set()
    k1 = len(rows[0])
    for r in rows:
        for i in range(k1):
            for j in range(i + 1, k1):
                used.add(r[j] - r[i])
    return used

def _greedy_random(n_rows, k, seed=0):
    rng = np.random.default_rng(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Randomly pick next x
            # To ensure progress, we can pick from a range
            # Or just increment x, but with random jumps?
            # Standard greedy: increment x until valid.
            # Randomized greedy: pick x from [current_max + 1, current_max + 50] randomly?
            # Let's stick to deterministic greedy for the initial solution to ensure validity,
            # but randomize the *order* of rows or starting offsets?
            # Actually, simple greedy is deterministic. 
            # To get diversity, we can add a random offset to the start of the search for each row?
            # No, they must start at 0.
            
            # Let's use a "randomized greedy" where we try a few random candidates 
            # and if none work, we fall back to linear scan.
            
            found = False
            # Try random candidates in [x+1, x+100]
            for _ in range(20):
                candidate = x + 1 + int(rng.integers(0, 100))
                nd = {candidate - a for a in row}
                if not (nd & used):
                    row.append(candidate)
                    used |= nd
                    x = candidate
                    found = True
                    break
            
            if not found:
                # Linear scan
                while True:
                    x += 1
                    nd = {x - a for a in row}
                    if not (nd & used):
                        row.append(x)
                        used |= nd
                        break
        
        rows.append(row)
    return rows
