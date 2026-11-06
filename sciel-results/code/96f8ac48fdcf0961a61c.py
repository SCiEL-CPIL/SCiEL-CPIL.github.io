import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n = 128
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # 0-indexed: positions 0 to 127
    progressions = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                progressions.append((a, b, c, e))
    
    num_progs = len(progressions)
    
    # For each position, list of progression indices it belongs to
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[e].append(idx)
    
    def is_valid(colors):
        """Check if coloring is valid: every 4-AP has at least 3 distinct colors."""
        for (a, b, c, e) in progressions:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                return False
        return True
    
    def count_violations(colors):
        """Count number of violated 4-APs."""
        v = 0
        for (a, b, c, e) in progressions:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                v += 1
        return v
    
    def try_backtracking(k, time_budget):
        """Try to find a valid k-coloring via backtracking with constraint propagation."""
        colors = [-1] * n
        # Progressions ending at position i: list of (a, b, c) where (a,b,c,i) is a 4-AP
        prog_ending_at = [[] for _ in range(n)]
        for (a, b, c, e) in progressions:
            prog_ending_at[e].append((a, b, c))
        
        def backtrack(i):
            if time.time() > time_budget:
                return False
            
            if i == n:
                return True
            
            # Try each color
            for c in range(k):
                valid = True
                # Check all 4-APs ending at position i
                for (a, b, c2) in prog_ending_at[i]:
                    cols = {colors[a], colors[b], colors[c2], c}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[i] = c
                if backtrack(i + 1):
                    return True
                colors[i] = -1
            
            return False
        
        # Symmetry breaking: fix colors[0] = 0
        colors[0] = 0
        
        # Try colors[1] = 0 first
        colors[1] = 0
        if backtrack(2):
            return colors[:]
        
        # Try colors[1] = 1
        if k >= 2:
            colors[1] = 1
            if backtrack(2):
                return colors[:]
        
        return None
    
    def local_search(k, max_iters, seed=42):
        """Local search to find valid k-coloring."""
        rng = random.Random(seed)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        
        # Track violations incrementally
        violated = [False] * num_progs
        total_violations = 0
        for idx, (a, b, c, e) in enumerate(progressions):
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                violated[idx] = True
                total_violations += 1
        
        if total_violations == 0:
            return colors[:]
        
        best_colors = colors[:]
        best_violations = total_violations
        no_improve_count = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random position
            i = rng.randint(0, n - 1)
            old_c = colors[i]
            
            # Find best new color for position i
            best_new_c = old_c
            best_delta = 0
            
            affected = prog_at[i]
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_viol = 0
                for idx in affected:
                    a, b, c2, e = progressions[idx]
                    if len({colors[a], colors[b], colors[c2], colors[e]}) < 3:
                        new_viol += 1
                
                # Current violations for affected progs
                cur_viol = sum(1 for idx in affected if violated[idx])
                delta = new_viol - cur_viol
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
            
            colors[i] = best_new_c
            
            # Update violation tracking
            if best_new_c != old_c:
                for idx in affected:
                    a, b, c2, e = progressions[idx]
                    old_v = violated[idx]
                    new_v = len({colors[a], colors[b], colors[c2], colors[e]}) < 3
                    if old_v != new_v:
                        if new_v:
                            total_violations += 1
                        else:
                            total_violations -= 1
                        violated[idx] = new_v
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    no_improve_count = 0
                    if best_violations == 0:
                        return best_colors
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Restart if stuck
            if no_improve_count > max_no_improve:
                # Random restart
                colors = [rng.randint(0, k - 1) for _ in range(n)]
                for idx, (a, b, c2, e) in enumerate(progressions):
                    violated[idx] = len({colors[a], colors[b], colors[c2], colors[e]}) < 3
                total_violations = sum(violated)
                no_improve_count = 0
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
        
        return best_colors if best_violations == 0 else None
    
    def greedy_construct(k):
        """Greedy left-to-right construction with backtracking."""
        colors = [-1] * n
        prog_ending_at = [[] for _ in range(n)]
        for (a, b, c, e) in progressions:
            prog_ending_at[e].append((a, b, c))
        
        def backtrack(i):
            if i == n:
                return True
            
            # Determine which colors are allowed at position i
            allowed = set(range(k))
            for (a, b, c2) in prog_ending_at[i]:
                existing = {colors[a], colors[b], colors[c2]}
                if len(existing) == 1:
                    # All same, need different
                    allowed.discard(existing.pop())
                elif len(existing) == 2:
                    # Two distinct, need a third
                    # If we pick one of the two existing, still only 2 colors -> invalid
                    # So we must pick a color NOT in existing
                    for c in list(allowed):
                        if c in existing:
                            allowed.discard(c)
                # If len(existing) == 3, any color works
            
            if not allowed:
                return False
            
            # Try colors in order (prefer smaller for symmetry)
            for c in sorted(allowed):
                colors[i] = c
                if backtrack(i + 1):
                    return True
            
            colors[i] = -1
            return False
        
        if backtrack(0):
            return colors[:]
        return None
    
    # Strategy:
    # 1. Try k=3 with backtracking (short time budget)
    # 2. Try k=3 with local search
    # 3. Try k=4 with backtracking
    # 4. Try k=4 with local search
    # 5. Try k=5 with local search
    # 6. Fallback to greedy with many colors
    
    # Try k=3 backtracking with 5s budget
    t1 = time.time()
    result = try_backtracking(3, t1 + 5.0)
    if result and is_valid(result):
        return result
    
    # Try k=3 local search
    t2 = time.time()
    result = local_search(3, 200000, seed=42)
    if result and is_valid(result):
        return result
    
    # Try k=3 local search with different seed
    result = local_search(3, 200000, seed=123)
    if result and is_valid(result):
        return result
    
    # Try k=4 backtracking with 5s budget
    t3 = time.time()
    result = try_backtracking(4, t3 + 5.0)
    if result and is_valid(result):
        return result
    
    # Try k=4 local search
    result = local_search(4, 200000, seed=42)
    if result and is_valid(result):
        return result
    
    # Try k=4 local search with different seed
    result = local_search(4, 200000, seed=123)
    if result and is_valid(result):
        return result
    
    # Try k=5 local search
    result = local_search(5, 200000, seed=42)
    if result and is_valid(result):
        return result
    
    # Try k=5 local search with different seed
    result = local_search(5, 200000, seed=123)
    if result and is_valid(result):
        return result
    
    # Fallback: greedy with many colors
    colors = []
    for x in range(1, 129):
        placed = False
        for color in range(128):
            ok = True
            # Check all 4-APs ending at position x (1-indexed)
            pos = x - 1  # 0-indexed
            for d in range(1, (x - 1) // 3 + 1):
                a = pos - 3 * d
                b = pos - 2 * d
                c = pos - d
                if a >= 0:
                    cols = {colors[a], colors[b], colors[c], color}
                    if len(cols) < 3:
                        ok = False
                        break
            if ok:
                colors.append(color)
                placed = True
                break
        if not placed:
            # Should not happen, but use a large color
            colors.append(127)
    
    # Verify and return
    if is_valid(colors):
        return colors
    
    # Last resort: use all distinct colors (always valid)
    return list(range(128))

# Ensure construct is defined (in case of import issues)
if 'construct' not in dir():
    def construct():
        return list(range(128))
