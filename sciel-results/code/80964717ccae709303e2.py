"""Rewritten program with stronger algorithm for 4-AP coloring."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    # Precompute all nonconstant 4-term progressions
    progressions = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            progressions.append((a, a + d, a + 2 * d, a + 3 * d))
    
    def count_distinct_colors(prog, colors):
        a, b, c, d_pos = prog
        return len({colors[a-1], colors[b-1], colors[c-1], colors[d_pos-1]})
    
    def check_valid(colors):
        """Check if a complete coloring is valid."""
        for prog in progressions:
            if count_distinct_colors(prog, colors) < 3:
                return False
        return True
    
    def try_greedy_with_k(k, seed_val):
        """Try greedy assignment with k colors."""
        rng = random.Random(seed_val)
        colors = [-1] * n
        
        # Assign positions in order 1 to n
        for pos in range(1, n + 1):
            # Determine which colors are allowed
            # A color c is allowed if for every progression containing pos,
            # assigning c doesn't make it impossible to have >=3 colors
            # Simple greedy: try colors, check if any completed progression violates
            
            valid_colors = set(range(k))
            
            # Check progressions where pos is the last element
            for prog in progressions:
                a, b, c, d_pos = prog
                if d_pos == pos:
                    # All previous are assigned
                    if colors[a-1] != -1 and colors[b-1] != -1 and colors[c-1] != -1:
                        # If all three previous are the same, we need a different color
                        prev_colors = {colors[a-1], colors[b-1], colors[c-1]}
                        if len(prev_colors) == 1:
                            # Must use a different color
                            valid_colors -= prev_colors
                        elif len(prev_colors) == 2:
                            # Can use any color, but if we use one of the two, still have 3
                            # Actually if prev has 2 colors, adding any color gives at least 2, 
                            # but we need at least 3. So we need a color NOT in prev_colors
                            valid_colors -= prev_colors
            
            # Also check progressions where pos is in the middle
            # This is more complex, so for simplicity, just check completed progressions
            # and use a simple heuristic
            
            if not valid_colors:
                return None
            
            # Pick the least used valid color
            color_counts = [0] * k
            for i in range(pos - 1):
                if colors[i] != -1:
                    color_counts[colors[i]] += 1
            
            best_color = min(valid_colors, key=lambda c: color_counts[c])
            colors[pos - 1] = best_color
        
        # Verify
        if check_valid(colors):
            return colors
        return None
    
    def try_backtracking(k, timeout):
        """Backtracking search with k colors."""
        deadline_local = time.time() + timeout
        colors = [-1] * n
        
        # Precompute for each position, which progressions it participates in
        pos_to_progs = [[] for _ in range(n + 1)]
        for idx, prog in enumerate(progressions):
            a, b, c, d_pos = prog
            pos_to_progs[a].append(idx)
            pos_to_progs[b].append(idx)
            pos_to_progs[c].append(idx)
            pos_to_progs[d_pos].append(idx)
        
        def is_consistent(pos, color):
            """Check if assigning color to position pos is consistent with all completed progressions."""
            colors[pos - 1] = color
            for prog_idx in pos_to_progs[pos]:
                a, b, c, d_pos = progressions[prog_idx]
                if (colors[a-1] != -1 and colors[b-1] != -1 and 
                    colors[c-1] != -1 and colors[d_pos-1] != -1):
                    if count_distinct_colors((a, b, c, d_pos), colors) < 3:
                        colors[pos - 1] = -1
                        return False
            colors[pos - 1] = -1
            return True
        
        def backtrack(pos):
            if time.time() > deadline_local:
                return False
            if pos > n:
                return True
            
            # Try colors in order
            for color in range(k):
                if is_consistent(pos, color):
                    colors[pos - 1] = color
                    if backtrack(pos + 1):
                        return True
                    colors[pos - 1] = -1
            
            return False
        
        if backtrack(1):
            return colors
        return None
    
    best_result = None
    best_color_count = n + 1
    
    # Try increasing number of colors
    for k in range(3, 8):
        if time.time() > deadline - 10:
            break
        
        # First try greedy (fast)
        for seed_val in range(20):
            if time.time() > deadline - 10:
                break
            result = try_greedy_with_k(k, seed_val)
            if result is not None:
                used = len(set(result))
                if used < best_color_count:
                    best_result = result
                    best_color_count = used
                break
        
        # Then try backtracking (slower but more thorough)
        if time.time() < deadline - 15:
            timeout = min(15, deadline - time.time() - 10)
            if timeout > 3:
                result = try_backtracking(k, timeout)
                if result is not None:
                    used = len(set(result))
                    if used < best_color_count:
                        best_result = result
                        best_color_count = used
        
        if best_result is not None and best_color_count == k:
            break  # Can't do better with this k
    
    if best_result is None:
        # Fallback: use a simple valid coloring with many colors
        # Assign each position a unique color (guaranteed valid)
        best_result = list(range(n))
        best_color_count = n
    
    # Ensure colors are in range 0..127
    color_set = set(best_result)
    if max(color_set) > 127 or min(color_set) < 0:
        mapping = {c: i for i, c in enumerate(sorted(color_set))}
        best_result = [mapping[c] for c in best_result]
    
    return best_result
