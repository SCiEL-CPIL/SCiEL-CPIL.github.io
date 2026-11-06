import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    # Precompute 4-term progressions: (a, a+d, a+2d, a+3d) with d > 0, all < n
    progressions = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e < n:
                progressions.append((a, b, c, e))
    
    num_progs = len(progressions)
    
    # For each position, list of (prog_idx, pos_in_prog) where pos_in_prog is 0,1,2,3
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(progressions):
        prog_at[a].append((idx, 0))
        prog_at[b].append((idx, 1))
        prog_at[c].append((idx, 2))
        prog_at[e].append((idx, 3))
    
    def count_distinct_colors(arr, idxs):
        """Count distinct colors at given indices."""
        seen = set()
        for i in idxs:
            seen.add(arr[i])
        return len(seen)
    
    def check_violation(colors, prog_idx):
        """Check if progression at prog_idx is violated (< 3 distinct colors)."""
        a, b, c, e = progressions[prog_idx]
        return count_distinct_colors(colors, [a, b, c, e]) < 3
    
    def total_violations(colors):
        v = 0
        for idx in range(num_progs):
            if check_violation(colors, idx):
                v += 1
        return v
    
    def solve_local_search(k, time_limit):
        """Local search to find valid k-coloring."""
        local_deadline = start_time + time_limit
        random.seed(42)
        
        # Multiple restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > local_deadline:
                break
            
            # Random initialization
            colors = [random.randint(0, k-1) for _ in range(n)]
            
            # Track violations for each progression
            violated = [check_violation(colors, i) for i in range(num_progs)]
            total_viol = sum(violated)
            
            if total_viol == 0:
                return colors
            
            # Local search
            no_improve_count = 0
            max_no_improve = 2000
            
            while time.time() < local_deadline and no_improve_count < max_no_improve:
                # Find a violated progression
                violated_indices = [i for i in range(num_progs) if violated[i]]
                if not violated_indices:
                    return colors
                
                # Pick a random violated progression
                p_idx = random.choice(violated_indices)
                a, b, c, e = progressions[p_idx]
                positions = [a, b, c, e]
                
                # Try changing one of the 4 positions
                best_improvement = 0
                best_change = None
                
                for pos in positions:
                    old_color = colors[pos]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        
                        colors[pos] = new_color
                        
                        # Recalculate violations for progressions involving pos
                        new_total_viol = total_viol
                        for (pidx, ppos) in prog_at[pos]:
                            old_viol = violated[pidx]
                            new_viol = check_violation(colors, pidx)
                            if old_viol != new_viol:
                                new_total_viol += (1 if new_viol else -1)
                                violated[pidx] = new_viol
                        
                        improvement = total_viol - new_total_viol
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_change = (pos, new_color)
                        
                        # Revert
                        colors[pos] = old_color
                        for (pidx, ppos) in prog_at[pos]:
                            # Restore original violation status
                            pass  # We'll handle this properly below
                
                if best_change is None:
                    # No improvement possible, try random moves (restart-ish)
                    no_improve_count += 1
                    # Make a random change to escape local minimum
                    pos = random.randint(0, n-1)
                    old_color = colors[pos]
                    new_color = random.randint(0, k-1)
                    while new_color == old_color:
                        new_color = random.randint(0, k-1)
                    colors[pos] = new_color
                    for (pidx, ppos) in prog_at[pos]:
                        violated[pidx] = check_violation(colors, pidx)
                    total_viol = sum(violated)
                    continue
                
                # Apply best change
                pos, new_color = best_change
                colors[pos] = new_color
                for (pidx, ppos) in prog_at[pos]:
                    violated[pidx] = check_violation(colors, pidx)
                total_viol = sum(violated)
                no_improve_count = 0
                
                if total_viol == 0:
                    return colors
        
        return None
    
    # Try k=3 first
    result = solve_local_search(3, 40.0)
    if result:
        return result
    
    if time.time() < deadline - 5.0:
        # Try k=4
        result = solve_local_search(4, 10.0)
        if result:
            return result
    
    # Fallback: greedy with backtracking
    progressions_ending_at = [[] for _ in range(n)]
    for (a, b, c, e) in progressions:
        progressions_ending_at[e].append((a, b, c))
    
    colors = [-1] * n
    colors[0] = 0
    
    def backtrack(i, k):
        if time.time() > deadline - 1.0:
            return False
        if i == n:
            return True
        
        for c in range(k):
            valid = True
            for (a, b, c_idx) in progressions_ending_at[i]:
                cols = {colors[a], colors[b], colors[c_idx], c}
                if len(cols) < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[i] = c
            if backtrack(i + 1, k):
                return True
            colors[i] = -1
        
        return False
    
    # Try with k=3
    if backtrack(1, 3):
        return colors
    
    # Try with k=4
    colors = [0] + [-1] * (n-1)
    if backtrack(1, 4):
        return colors
    
    # Final fallback: return any valid coloring (shouldn't happen)
    return list(range(n))
