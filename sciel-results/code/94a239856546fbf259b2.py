import numpy as np
from itertools import combinations
import time
import random

def construct():
    """
    Construct a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses backtracking with efficient C4 checking.
    """
    
    def solve(n, timeout=110):
        start_time = time.time()
        colors = [[-1]*n for _ in range(n)]
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # cn[c][i][j] = number of x (x!=i,j) with color(x,i)==c and color(x,j)==c
        cn = [[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(5)]
        
        def get_c(i, j):
            if i > j:
                i, j = j, i
            return colors[i][j]
        
        def update_cn_on_assign(i, j, c):
            """Update cn after assigning color c to edge (i,j)."""
            # When we set color(i,j)=c, for any x (x!=i,j):
            # If color(x,i)==c, then cn[c][x][j] increases by 1 (x is now a common neighbor of i and j in color c? No)
            # Actually: cn[c][a][b] = |{x: x!=a,b, color(x,a)==c, color(x,b)==c}|
            # Setting color(i,j)=c affects cn[c][x][i] and cn[c][x][j] for x where the other edge is c.
            # Specifically: for any x (x!=i,j):
            #   If color(x,i)==c: then cn[c][x][j] should increase by 1? No, cn[c][x][j] counts y with color(y,x)==c and color(y,j)==c. 
            #   We're setting color(i,j)=c, so if color(x,i)==c, then i is a common neighbor of x and j in color c.
            #   So cn[c][x][j] increases by 1 if color(x,i)==c.
            #   Similarly, cn[c][j][x] increases by 1 if color(x,i)==c.
            #   If color(x,j)==c: then cn[c][x][i] increases by 1.
            #   And cn[c][i][x] increases by 1.
            # Also, cn[c][i][j] is not affected by this assignment (it counts x, not i or j).
            
            for x in range(n):
                if x == i or x == j:
                    continue
                ci = get_c(x, i)
                cj = get_c(x, j)
                if ci == c:
                    cn[c][x][j] += 1
                    cn[c][j][x] += 1
                if cj == c:
                    cn[c][x][i] += 1
                    cn[c][i][x] += 1
        
        def undo_cn_on_assign(i, j, c):
            """Undo cn update after unassigning color c from edge (i,j)."""
            for x in range(n):
                if x == i or x == j:
                    continue
                ci = get_c(x, i)
                cj = get_c(x, j)
                if ci == c:
                    cn[c][x][j] -= 1
                    cn[c][j][x] -= 1
                if cj == c:
                    cn[c][x][i] -= 1
                    cn[c][i][x] -= 1
        
        def backtrack(idx):
            if time.time() - start_time > timeout:
                return False
            if idx == len(edges):
                return True
            
            i, j = edges[idx]
            for c in range(5):
                if cn[c][i][j] < 2:
                    colors[i][j] = c
                    update_cn_on_assign(i, j, c)
                    if backtrack(idx + 1):
                        return True
                    undo_cn_on_assign(i, j, c)
                    colors[i][j] = -1
            
            return False
        
        return backtrack(0)
    
    # Try decreasing n until we find a solution
    for target_n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if solve(target_n, timeout=110):
            colors_list = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    colors_list.append(int(colors[i][j]))
            return {'n': target_n, 'colors': colors_list}
    
    return {'n': 1, 'colors': []}
