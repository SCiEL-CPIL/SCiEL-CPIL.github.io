import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    N = 128
    
    # Try k=3
    for k in [3, 4]:
        colors = [-1] * (N + 1)
        
        # Fix c[1] = 0
        colors[1] = 0
        start_idx = 2
        
        # Precompute d ranges for efficiency
        # end_ds[i] = list of d such that i-3d >= 1
        # future_ds[i] = list of d such that i-2d >= 1 and i+3d <= N
        end_ds = [[] for _ in range(N + 1)]
        future_ds = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            for d in range(1, (i - 1) // 3 + 1):
                end_ds[i].append(d)
            for d in range(1, min((i - 1) // 2, (N - i) // 3) + 1):
                future_ds[i].append(d)

        def is_valid(i, c):
            # Check past APs
            for d in end_ds[i]:
                x1 = i - 3 * d
                x2 = i - 2 * d
                x3 = i - d
                c1 = colors[x1]
                c2 = colors[x2]
                c3 = colors[x3]
                
                if c1 == c2 == c3:
                    return False
                if c1 == c2 and c2 != c3:
                    # Colors: a, a, b. Need c4 != a, c4 != b.
                    if c == c1 or c == c3:
                        return False
                elif c1 == c3 and c2 != c1:
                    # Colors: a, b, a. Need c4 != a, c4 != b.
                    if c == c1 or c == c2:
                        return False
                elif c2 == c3 and c1 != c2:
                    # Colors: a, b, b. Need c4 != a, c4 != b.
                    if c == c1 or c == c2:
                        return False
                elif c1 != c2 and c2 != c3 and c1 != c3:
                    # 3 distinct colors. Any c is ok.
                    pass
                else:
                    # This case is covered by above?
                    # c1=c2=c3 handled.
                    # Pairs handled.
                    # All distinct handled.
                    pass
            
            # Check future triples
            for d in future_ds[i]:
                if colors[i - 2 * d] == c and colors[i - d] == c:
                    return False
            
            return True

        def solve(i):
            if time.time() > deadline:
                return False
            
            if i > N:
                return True
            
            for c in range(k):
                if is_valid(i, c):
                    colors[i] = c
                    if solve(i + 1):
                        return True
                    colors[i] = -1
            return False

        if solve(start_idx):
            return colors[1:]
            
    # Fallback: return a valid coloring with many colors
    # Use distinct colors to be safe
    return list(range(128))
