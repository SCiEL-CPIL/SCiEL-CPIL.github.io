import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Base 32-vertex coloring
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
        
    def get_16_color(i, j):
        return log[(i % 16) ^ (j % 16)] % 3

    def make_32_coloring():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(get_16_color(i, j))
                else:
                    colors.append(3)
        return colors

    best_colors = make_32_coloring()
    best_n = 32
    
    # Try to extend to 33, 34, ... using greedy search
    # This is a heuristic search.
    
    def is_valid(colors, n):
        # Check for monochromatic triangles
        # This is O(n^3) which is too slow for n=50 (125k checks, ok)
        # But we need to be fast.
        # Use bitsets?
        # For now, simple check.
        # Map edge to color
        # adj[i][j] = color
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        return False
        return True

    def try_extend(colors, n, target_n):
        # Greedy addition of vertices
        # This is very slow in Python.
        # Skip for now.
        pass

    return {'n': best_n, 'colors': best_colors}
