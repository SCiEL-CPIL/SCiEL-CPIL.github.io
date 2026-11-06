import time
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for overhead and return
    
    def solve(n):
        if n < 2:
            return [0] * (n * (n - 1) // 2)
        
        # Edge index mapping
        # (i, j) with i < j -> index
        # We can use a 2D array for colors, -1 if unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # List of edges in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Symmetry breaking: fix first few edges from vertex 0
        # This is valid because we can permute colors 0,1,2,3
        # and permute vertices.
        # Fixing C[0,1]=0, C[0,2]=1, C[0,3]=2, C[0,4]=3 is a strong constraint.
        # This assumes n >= 5.
        if n >= 5:
            colors[0][1] = colors[1][0] = 0
            colors[0][2] = colors[2][0] = 1
            colors[0][3] = colors[3][0] = 2
            colors[0][4] = colors[4][0] = 3
        
        # Recursive backtracking
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                raise TimeoutError
            
            if idx == len(edges):
                return True
            
            i, j = edges[idx]
            
            # If already assigned (due to symmetry breaking), move to next
            if colors[i][j] != -1:
                return backtrack(idx + 1)
            
            for c in range(4):
                # Check if assigning c to (i, j) creates a monochromatic triangle
                valid = True
                # Check k < i
                for k in range(i):
                    if colors[i][k] == c and colors[j][k] == c:
                        valid = False
                        break
                if valid:
                    # Check i < k < j
                    for k in range(i + 1, j):
                        if colors[i][k] == c and colors[k][j] == c:
                            valid = False
                            break
                
                if valid:
                    colors[i][j] = c
                    colors[j][i] = c
                    if backtrack(idx + 1):
                        return True
                    colors[i][j] = -1
                    colors[j][i] = -1
            return False
        
        try:
            if backtrack(0):
                # Convert to flat list in lexicographic order
                res = []
                for i in range(n):
                    for j in range(i + 1, n):
                        res.append(colors[i][j])
                return res
        except TimeoutError:
            return None
        return None

    # Try n from 50 downwards
    # We know 32 is valid. 50 is the record.
    # We will try 50, 49, ... until we find one or time out.
    # However, if 50 times out, we might not have time for 49.
    # So we should allocate time wisely.
    
    # Let's try 50 with a sub-limit? No, just try 50. If it fails, try 40 (easier) to get a score > 1.
    # Actually, if 50 fails, the best fallback is a known valid one.
    # I will generate a known valid 32-vertex coloring as a fallback.
    
    def get_fallback():
        # The seed program's 32-vertex coloring
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return 32, colors

    best_n, best_colors = get_fallback()
    
    # Try to find better
    # We can try 50, then 45, then 40 if time permits.
    # But backtracking is exponential.
    # Let's just try 50. If it succeeds, great.
    # If it fails, we return the fallback.
    
    # To be safe, let's try 50 with a tight time limit inside solve?
    # The solve function checks time.
    
    res_50 = solve(50)
    if res_50 is not None:
        return {'n': 50, 'colors': res_50}
    
    # If 50 failed, try 40?
    # 40 is much easier.
    if time.time() - start_time < time_limit - 10:
        res_40 = solve(40)
        if res_40 is not None:
            return {'n': 40, 'colors': res_40}
            
    return {'n': best_n, 'colors': best_colors}
