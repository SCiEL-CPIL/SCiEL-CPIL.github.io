import time

def construct():
    deadline = time.time() + 85
    N = 128
    colors = [-1] * N
    
    def is_valid_assignment(pos, c):
        """Check if assigning color c to position pos maintains validity."""
        colors[pos] = c
        ok = True
        
        # Check all 4-term APs where pos is involved and all 4 positions are assigned
        # pos as 1st: (pos, pos+d, pos+2d, pos+3d)
        for d in range(1, (N - pos) // 3 + 1):
            p1, p2, p3 = pos + d, pos + 2 * d, pos + 3 * d
            if p3 < N and colors[p1] != -1 and colors[p2] != -1 and colors[p3] != -1:
                if len({colors[pos], colors[p1], colors[p2], colors[p3]}) < 3:
                    ok = False
                    break
        if not ok:
            colors[pos] = -1
            return False
        
        # pos as 2nd: (pos-d, pos, pos+d, pos+2d)
        for d in range(1, pos + 1):
            a, p3, p4 = pos - d, pos + d, pos + 2 * d
            if a >= 0 and p3 < N and p4 < N and colors[a] != -1 and colors[p3] != -1 and colors[p4] != -1:
                if len({colors[a], colors[pos], colors[p3], colors[p4]}) < 3:
                    ok = False
                    break
        if not ok:
            colors[pos] = -1
            return False
        
        # pos as 3rd: (pos-2d, pos-d, pos, pos+d)
        for d in range(1, (pos + 1) // 2 + 1):
            a, p2, p4 = pos - 2 * d, pos - d, pos + d
            if a >= 0 and p2 >= 0 and p4 < N and colors[a] != -1 and colors[p2] != -1 and colors[p4] != -1:
                if len({colors[a], colors[p2], colors[pos], colors[p4]}) < 3:
                    ok = False
                    break
        if not ok:
            colors[pos] = -1
            return False
        
        # pos as 4th: (pos-3d, pos-2d, pos-d, pos)
        for d in range(1, (pos + 1) // 3 + 1):
            a, p2, p3 = pos - 3 * d, pos - 2 * d, pos - d
            if a >= 0 and p2 >= 0 and p3 >= 0 and colors[a] != -1 and colors[p2] != -1 and colors[p3] != -1:
                if len({colors[a], colors[p2], colors[p3], colors[pos]}) < 3:
                    ok = False
                    break
        if not ok:
            colors[pos] = -1
            return False
        
        return True
    
    for i in range(N):
        for c in range(8):
            if is_valid_assignment(i, c):
                break
        else:
            colors[i] = 0  # Fallback
    
    return colors
