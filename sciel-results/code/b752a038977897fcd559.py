def construct():
    n = 128
    
    def valid_coloring(colors):
        """Check if coloring is valid: every 4-term AP has >= 3 distinct colors"""
        for a in range(1, n + 1):
            for d in range(1, (n - a) // 3 + 1):
                c = [colors[a-1], colors[a+d-1], colors[a+2*d-1], colors[a+3*d-1]]
                if len(set(c)) < 3:
                    return False
        return True
    
    def check_all_aps_involving(colors, pos, color):
        """Check all 4-term APs that include position pos (1-indexed) and have all other positions assigned.
        Returns True if no violation."""
        # pos is 4th term: (pos-3d, pos-2d, pos-d, pos)
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                break
            c = [colors[a-1], colors[a+d-1], colors[a+2*d-1], color]
            if len(set(c)) < 3:
                return False
        
        # pos is 3rd term: (pos-2d, pos-d, pos, pos+d) — need pos+d <= n and assigned
        for d in range(1, (pos - 1) // 2 + 1):
            a = pos - 2 * d
            if a < 1:
                break
            fourth = pos + d
            if fourth > n:
                continue
            if colors[fourth - 1] == -1:
                continue  # not yet assigned, can't check
            c = [colors[a-1], colors[a+d-1], color, colors[fourth-1]]
            if len(set(c)) < 3:
                return False
        
        # pos is 2nd term: (pos-d, pos, pos+d, pos+2d)
        for d in range(1, (n - pos) // 2 + 1):
            a = pos - d
            if a < 1:
                break
            third = pos + d
            fourth = pos + 2 * d
            if fourth > n:
                continue
            if colors[a-1] == -1 or colors[third-1] == -1 or colors[fourth-1] == -1:
                continue
            c = [colors[a-1], color, colors[third-1], colors[fourth-1]]
            if len(set(c)) < 3:
                return False
        
        # pos is 1st term: (pos, pos+d, pos+2d, pos+3d)
        for d in range(1, (n - pos) // 3 + 1):
            second = pos + d
            third = pos + 2 * d
            fourth = pos + 3 * d
            if fourth > n:
                break
            if colors[second-1] == -1 or colors[third-1] == -1 or colors[fourth-1] == -1:
                continue
            c = [color, colors[second-1], colors[third-1], colors[fourth-1]]
            if len(set(c)) < 3:
                return False
        
        return True
    
    # Try backtracking with 3 colors
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(pos, colors):
        if pos > n:
            return True
        
        # Try each color
        for c in range(3):
            if check_all_aps_involving(colors, pos, c):
                colors[pos - 1] = c
                if backtrack(pos + 1, colors):
                    return True
                colors[pos - 1] = -1
        
        return False
    
    colors = [-1] * n
    if backtrack(1, colors):
        return colors
    
    # Fallback: try with 4 colors
    def backtrack4(pos, colors):
        if pos > n:
            return True
        
        for c in range(4):
            if check_all_aps_involving(colors, pos, c):
                colors[pos - 1] = c
                if backtrack4(pos + 1, colors):
                    return True
                colors[pos - 1] = -1
        
        return False
    
    colors = [-1] * n
    if backtrack4(1, colors):
        return colors
    
    # Last resort: use many colors, one per position
    return list(range(n))
