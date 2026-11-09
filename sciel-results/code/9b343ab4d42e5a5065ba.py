import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    
    # Precompute constraints
    constraints = [[] for _ in range(N)]
    for x in range(N):
        pos = x + 1
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                continue
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[x].append((i1, i2, i3))

    affected = [[] for _ in range(N)]
    for y in range(N):
        for i1, i2, i3 in constraints[y]:
            affected[i1].append(y)
            affected[i2].append(y)
            affected[i3].append(y)

    def is_valid_assignment(colors, x, c):
        for i1, i2, i3 in constraints[x]:
            c1, c2, c3 = colors[i1], colors[i2], colors[i3]
            if c1 is None or c2 is None or c3 is None:
                continue
            if len({c1, c2, c3, c}) < 3:
                return False
        return True

    def forward_check(colors, x, c, k):
        for y in affected[x]:
            if colors[y] is not None:
                continue
            has_valid = False
            for cy in range(k):
                if is_valid_assignment(colors, y, cy):
                    has_valid = True
                    break
            if not has_valid:
                return False
        return True

    def backtrack(x, colors, k):
        if time.time() > deadline:
            return None
            
        if x == N:
            return colors[:]
            
        max_color_to_try = min(k, x + 1)
        
        valid_colors = []
        for c in range(max_color_to_try):
            if is_valid_assignment(colors, x, c):
                valid_colors.append(c)
                
        if not valid_colors:
            return None

        used_colors = set(colors[:x])
        valid_colors.sort(key=lambda c: (0 if c in used_colors else 1, c))
        
        for c in valid_colors:
            colors[x] = c
            if forward_check(colors, x, c, k):
                res = backtrack(x + 1, colors, k)
                if res is not None:
                    return res
            colors[x] = None
            
        return None

    # Try k=3, 4, 5
    for k in [3, 4, 5]:
        colors = [None] * N
        colors[0] = 0
        res = backtrack(1, colors, k)
        if res is not None:
            return res

    # Fallback
    random.seed(0)
    colors = [None] * N
    for x in range(N):
        valid_colors = []
        for c in range(128):
            if is_valid_assignment(colors, x, c):
                valid_colors.append(c)
        if not valid_colors:
            break
        used = set(colors[:x])
        used_valid = [c for c in valid_colors if c in used]
        if used_valid:
            colors[x] = random.choice(used_valid)
        else:
            colors[x] = valid_colors[0]
    return colors
