import time
import sys
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    
    # Precompute 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0
    # Indices are 0-based in our array.
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
    
    # For each index i, store the list of progressions it belongs to.
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    # For each index i, store progressions where i is the last element (d).
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, d) in progressions:
        progressions_ending_at[d].append((a, b, c))

    def count_violations(colors):
        v = 0
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                v += 1
        return v

    def is_valid(colors):
        return count_violations(colors) == 0

    # Strategy:
    # 1. Try to find a valid coloring with k=3 using backtracking with constraint propagation.
    # 2. If that fails, try k=4.
    # 3. If that fails, use local search (WalkSAT style) with k=3, then k=4, etc.
    # 4. Fallback to a greedy construction that uses more colors but is guaranteed to work.

    def solve_backtracking(k, time_limit):
        colors = [-1] * n
        colors[0] = 0
        
        # Symmetry breaking:
        # We can fix colors[0] = 0.
        # We can fix colors[1] to be 0 or 1 (if k>=2).
        # More generally, we can use the "first occurrence" symmetry breaking:
        # The first time a new color appears, it must be the next unused color index.
        
        used_colors = {0}
        
        def backtrack(i):
            if time.time() > time_limit:
                return False
            
            if i == n:
                return True
            
            # Determine available colors for position i
            # Use MRV heuristic? Or just try 0..k-1?
            # Let's try to restrict the domain based on constraints.
            
            # For now, just try all k colors, but with a heuristic order.
            # Heuristic: Try colors that are already used first? Or least used?
            # Let's just try 0..k-1.
            
            # Check if any color is allowed
            allowed = []
            for c in range(k):
                # Check backward constraints (progressions ending at i)
                valid = True
                for (a, b, c_idx) in progressions_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c_idx], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if valid:
                    # Forward Checking:
                    # Check if this assignment makes any future progression unsatisfiable.
                    # For each progression (a, b, c, d) where i is a, b, or c:
                    # If the other two of {a,b,c} are assigned, check if a valid color exists for d.
                    # If {A,B,C} has size 1, D must be != A. (Requires k>=2)
                    # If {A,B,C} has size 2, D must be a 3rd color. (Requires k>=3)
                    # If {A,B,C} has size 3, D can be anything.
                    
                    # This check is weak but might help.
                    # Let's skip it for now to keep it simple and fast.
                    allowed.append(c)
            
            if not allowed:
                return False
            
            # Try allowed colors
            # Order: Maybe try colors that appear less often in the prefix?
            # For now, just random or fixed order.
            for c in allowed:
                colors[i] = c
                if backtrack(i + 1):
                    return True
                colors[i] = -1
                
            return False
        
        # Try colors[1] = 0
        colors[1] = 0
        if backtrack(2):
            return colors[:]
        
        # Try colors[1] = 1
        if k >= 2:
            colors[1] = 1
            if backtrack(2):
                return colors[:]
                
        return None

    def solve_local_search(k, time_limit, seed=42):
        random.seed(seed)
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        # Precompute violation status for each progression
        is_violated = [False] * len(progressions)
        for idx, (a, b, c, d) in enumerate(progressions):
            is_violated[idx] = len({colors[a], colors[b], colors[c], colors[d]}) < 3
            
        violations = sum(is_violated)
        
        if violations == 0:
            return colors[:]
            
        best_colors = colors[:]
        best_violations = violations
        
        for _ in range(200000):  # Max iterations
            if time.time() > time_limit:
                break
                
            # Pick a random position
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            # Find best new color for i
            best_new_c = old_c
            best_delta = 0
            
            # Calculate delta for each new color
            # Only progressions containing i are affected
            affected_progs = prog_at[i]
            
            # Current violations for affected progs
            current_violations = sum(1 for idx in affected_progs if is_violated[idx])
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_violations = 0
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    if len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3:
                        new_violations += 1
                        
                delta = new_violations - current_violations
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            colors[i] = best_new_c
            
            # Update is_violated and violations count
            if best_new_c != old_c:
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    old_violated = is_violated[idx]
                    new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                    if old_violated != new_violated:
                        if new_violated:
                            violations += 1
                        else:
                            violations -= 1
                        is_violated[idx] = new_violated
                        
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors[:]
                        
        return best_colors[:] if best_violations == 0 else None

    # Try k=3 with backtracking first (with a short time limit)
    result = solve_backtracking(3, start_time + 5.0)
    if result:
        return result
        
    # Try k=3 with local search
    result = solve_local_search(3, start_time + 10.0)
    if result:
        return result
        
    # Try k=4 with backtracking
    result = solve_backtracking(4, start_time + 15.0)
    if result:
        return result
        
    # Try k=4 with local search
    result = solve_local_search(4, start_time + 25.0)
    if result:
        return result
        
    # Try k=5 with local search
    result = solve_local_search(5, start_time + 35.0)
    if result:
        return result

    # Fallback: Greedy construction
    # This might use more colors, but should always find a valid coloring.
    colors = []
    for x in range(n):
        # Try to find a color for position x (0-indexed)
        found = False
        for c in range(128):  # Max 128 colors
            valid = True
            # Check all 4-term progressions ending at x
            for (a, b, c_idx) in progressions_ending_at[x]:
                cols = {colors[a], colors[b], colors[c_idx], c}
                if len(cols) < 3:
                    valid = False
                    break
            if valid:
                # Also need to check progressions where x is not the last element?
                # No, if we assign in order, we only need to check progressions where x is the last element.
                # Because other progressions involving x will be checked when their last element is assigned.
                colors.append(c)
                found = True
                break
        if not found:
            # This should not happen if we have enough colors.
            # But just in case, assign a new color.
            colors.append(len(set(colors)))
            
    return colors

# Run construct
if __name__ == "__main__":
    result = construct()
    # Verify
    n = 128
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
    
    valid = True
    for (a, b, c, d) in progressions:
        if len({result[a], result[b], result[c], result[d]}) < 3:
            valid = False
            break
    
    num_colors = len(set(result))
    print(f"Valid: {valid}, Num Colors: {num_colors}")
    print(result[:20])
