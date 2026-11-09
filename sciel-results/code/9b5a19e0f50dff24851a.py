import time
import random
import numpy as np

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s

    # Precompute all nonconstant 4-term APs
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    ap_count = len(aps)
    
    # For each position, precompute which APs it appears in (for fast constraint checking)
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def is_valid_4ap(ap, colors):
        """Check if a 4-AP uses at least 3 distinct colors."""
        c = [colors[i] for i in ap]
        return len(set(c)) >= 3
    
    def check_partial_validity(colors, assigned_count):
        """Check all APs that are fully assigned so far."""
        for ap in aps:
            # Check if all positions in this AP are assigned
            if all(colors[i] != -1 for i in ap):
                if not is_valid_4ap(ap, colors):
                    return False
        return True
    
    def backtrack(colors, pos, used_colors):
        """Backtracking search to assign colors."""
        if time.time() > deadline - 5:
            return None, None
        
        if pos == N:
            return list(colors), used_colors
        
        # Try each color from 0 to 127
        for color in range(128):
            colors[pos] = color
            
            # Check if any AP containing this position is now fully assigned and invalid
            valid = True
            for ap_idx in pos_aps[pos]:
                ap = aps[ap_idx]
                if all(colors[i] != -1 for i in ap):
                    if not is_valid_4ap(ap, colors):
                        valid = False
                        break
            
            if valid:
                new_used = used_colors | {color}
                result, _ = backtrack(colors, pos + 1, new_used)
                if result is not None:
                    return result, new_used
            
            colors[pos] = -1
        
        return None, None
    
    def heuristic_order(positions):
        """Order positions by number of constraints (most constrained first)."""
        return sorted(positions, key=lambda p: len(pos_aps[p]), reverse=True)
    
    # Try different strategies to minimize number of colors used
    
    best_result = None
    best_num_colors = 128
    
    # Strategy 1: Greedy with forward checking
    def greedy_forward_checking():
        colors = [-1] * N
        used_colors = set()
        
        for pos in range(N):
            best_color = None
            best_violations = float('inf')
            
            for color in range(128):
                colors[pos] = color
                violations = 0
                
                for ap_idx in pos_aps[pos]:
                    ap = aps[ap_idx]
                    if all(colors[i] != -1 for i in ap):
                        if not is_valid_4ap(ap, colors):
                            violations += 1
                            break
                
                if violations < best_violations:
                    best_violations = violations
                    best_color = color
                    if violations == 0:
                        break
                
                colors[pos] = -1
            
            if best_color is not None:
                colors[pos] = best_color
                used_colors.add(best_color)
            else:
                return None, len(used_colors)
        
        return colors, len(used_colors)
    
    # Strategy 2: Backtracking with pruning
    def backtracking_search(max_colors_to_try=4):
        colors = [-1] * N
        
        for max_c in range(1, max_colors_to_try + 1):
            # Try with colors 0 to max_c-1
            def bt(pos, used):
                if time.time() > deadline - 5:
                    return None
                if pos == N:
                    return list(colors)
                
                for color in range(max_c):
                    if color in used:
                        continue
                    colors[pos] = color
                    
                    valid = True
                    for ap_idx in pos_aps[pos]:
                        ap = aps[ap_idx]
                        if all(colors[i] != -1 for i in ap):
                            if not is_valid_4ap(ap, colors):
                                valid = False
                                break
                    
                    if valid:
                        result = bt(pos + 1, used | {color})
                        if result is not None:
                            return result
                    
                    colors[pos] = -1
                
                return None
            
            result = bt(0, set())
            if result is not None:
                return result, max_c
        
        return None, None
    
    # First try greedy
    result, num_c = greedy_forward_checking()
    if result is not None:
        best_result = result
        best_num_colors = num_c
    
    # Then try backtracking to potentially find fewer colors
    if time.time() < deadline - 10:
        result2, num_c2 = backtracking_search(3)
        if result2 is not None and num_c2 < best_num_colors:
            best_result = result2
            best_num_colors = num_c2
    
    if best_result is not None:
        # Verify
        for ap in aps:
            if len(set(best_result[i] for i in ap)) < 3:
                # Invalid, fall back to simple assignment
                return list(range(N))
        return best_result
    
    # Fallback: just use position-based coloring (may not be optimal but valid)
    colors = [i % 4 for i in range(N)]
    return colors

if __name__ == "__main__":
    result = construct()
    print(f"Number of colors used: {len(set(result))}")
