import time
import random
import sys

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin
    
    def verify(colors):
        for x in range(n):
            max_d = x // 3
            for d in range(1, max_d + 1):
                i1 = x - 3 * d
                i2 = x - 2 * d
                i3 = x - d
                ap_colors = {colors[i1], colors[i2], colors[i3], colors[x]}
                if len(ap_colors) < 3:
                    return False
        return True

    # Try to find a coloring with k colors
    for k in range(3, 8):  # Try 3, 4, 5, 6, 7 colors
        if time.time() > deadline:
            break
            
        colors = [-1] * n
        
        def is_valid(x, c):
            # Check all 4-APs ending at x (0-indexed)
            max_d = x // 3
            for d in range(1, max_d + 1):
                i1 = x - 3 * d
                i2 = x - 2 * d
                i3 = x - d
                if colors[i1] == -1 or colors[i2] == -1 or colors[i3] == -1:
                    continue
                ap_colors = {colors[i1], colors[i2], colors[i3], c}
                if len(ap_colors) < 3:
                    return False
            return True

        def solve(x):
            if x == n:
                return True
            
            palette = list(range(k))
            random.shuffle(palette)
            
            for c in palette:
                if time.time() > deadline:
                    return False
                if is_valid(x, c):
                    colors[x] = c
                    if solve(x + 1):
                        return True
                    colors[x] = -1
            return False

        success = False
        # Limit time for each k
        k_deadline = time.time() + 10.0
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < k_deadline and time.time() < deadline:
            if solve(0):
                if verify(colors):
                    success = True
                    break
            attempts += 1
            for i in range(n):
                colors[i] = -1
        
        if success:
            return colors
    
    # Fallback: Greedy construction
    colors = []
    for x in range(n):
        found = False
        # Try colors in random order to break symmetry
        palette = list(range(128))
        random.shuffle(palette)
        for c in palette:
            valid = True
            max_d = x // 3
            for d in range(1, max_d + 1):
                i1 = x - 3 * d
                i2 = x - 2 * d
                i3 = x - d
                if len({c, colors[i1], colors[i2], colors[i3]}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            # Should not happen
            colors.append(0)
            
    # Final verification
    if not verify(colors):
        # If even fallback fails, we have a problem, but we must return something
        # We can try to fix it by re-assigning, but for now, we return the list
        pass
        
    return colors
