import time

def construct():
    n = 128
    colors = [-1] * n
    start_time = time.time()
    deadline = start_time + 50.0

    def is_valid_assignment(idx, c):
        # Check all 4-term APs ending at idx
        # idx = a + 3d (0-indexed relative to start of AP)
        # Actually, if idx is the 4th term, the terms are idx-3d, idx-2d, idx-d, idx.
        # We need idx-3d >= 0.
        for d in range(1, idx // 3 + 1):
            i3 = idx - 3 * d
            i2 = idx - 2 * d
            i1 = idx - d
            i0 = idx
            
            c3 = colors[i3]
            c2 = colors[i2]
            c1 = colors[i1]
            c0 = c
            
            # All previous indices are assigned
            distinct = len({c0, c1, c2, c3})
            if distinct < 3:
                return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        # Try colors 0, 1, 2
        for c in range(3):
            if is_valid_assignment(idx, c):
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
        return False

    if backtrack(0):
        return colors
    else:
        # Fallback: Greedy
        colors = [-1] * n
        for i in range(n):
            for c in range(3):
                if is_valid_assignment(i, c):
                    colors[i] = c
                    break
        return colors
