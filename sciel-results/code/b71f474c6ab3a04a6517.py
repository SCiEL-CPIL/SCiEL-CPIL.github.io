import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 128
    
    # Precompute constraints for each index
    # constraints[i] is a list of d values such that the AP ending at i (0-based)
    # is (i-3d, i-2d, i-d, i)
    constraints = [[] for _ in range(N)]
    for i in range(N):
        max_d = i // 3
        for d in range(1, max_d + 1):
            constraints[i].append(d)
            
    def solve(k):
        """
        Try to find a coloring of length N using colors 0..k-1.
        Returns the list of colors or None.
        """
        colors = [-1] * N
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == N:
                return True
            
            # Try each color
            for val in range(k):
                colors[idx] = val
                
                valid = True
                for d in constraints[idx]:
                    a = idx - 3 * d
                    c1 = colors[a]
                    c2 = colors[a + d]
                    c3 = colors[a + 2 * d]
                    c4 = val
                    
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                colors[idx] = -1
            
            return False
        
        if backtrack(0):
            return colors
        return None

    # Minimum colors must be at least 3
    sol = solve(3)
    if sol is not None:
        return sol
        
    # If k=3 fails, try k=4
    if time.time() < deadline:
        sol = solve(4)
        if sol is not None:
            return sol
            
    # Fallback: Greedy approach
    # This ensures we return a valid coloring if backtracking times out or fails
    colors = []
    for x in range(1, N + 1):
        for c in range(128):
            ok = True
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                a = x - 1 - 3 * d
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                c4 = c
                if len({c1, c2, c3, c4}) < 3:
                    ok = False
                    break
            if ok:
                colors.append(c)
                break
        else:
            colors.append(0)
            
    return colors
