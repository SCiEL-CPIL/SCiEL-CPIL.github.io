import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Lazebnik-Woldar construction for n=26
    # 5 caps in Z_5^2, each C4-free
    
    S = [
        [(1,0), (4,0), (0,1), (0,4)],
        [(1,1), (4,4), (1,4), (4,1)],
        [(1,2), (4,3), (2,1), (3,4)],
        [(1,3), (4,2), (3,1), (2,4)],
        [(2,2), (3,3), (2,3), (3,2)]
    ]
    
    color_of_diff = {}
    for c in range(5):
        for vec in S[c]:
            color_of_diff[vec] = c
    
    n = 26
    colors = []
    
    for i in range(25):
        for j in range(i+1, 25):
            vi = (i//5, i%5)
            vj = (j//5, j%5)
            diff = ((vj[0]-vi[0])%5, (vj[1]-vi[1])%5)
            colors.append(color_of_diff.get(diff, 0))
    
    for j in range(25):
        colors.append(j % 5)
    
    def is_valid(col_list, n):
        neighbors = [[[] for _ in range(n)] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = col_list[idx]
                neighbors[c][i].append(j)
                neighbors[c][j].append(i)
                idx += 1
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    if len(set(neighbors[c][u]) & set(neighbors[c][v])) >= 2:
                        return False
        return True
    
    if not is_valid(colors, n):
        # Local search to fix
        def count_bad(col_list):
            neighbors = [[[] for _ in range(n)] for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = col_list[idx]
                    neighbors[c][i].append(j)
                    neighbors[c][j].append(i)
                    idx += 1
            bad = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u+1, n):
                        cnt = len(set(neighbors[c][u]) & set(neighbors[c][v]))
                        if cnt >= 2:
                            bad += cnt * (cnt-1) // 2
            return bad
        
        best_colors = list(colors)
        best_bad = count_bad(colors)
        T = 1.0
        time_limit = deadline - 5
        while time.time() < time_limit and best_bad > 0:
            j = random.randint(0, 24)
            idx_25 = 300 + j
            old_c = colors[idx_25]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            colors[idx_25] = new_c
            new_bad = count_bad(colors)
            if new_bad <= count_bad(best_colors) or random.random() < math.exp(-(new_bad - best_bad)/T):
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_colors = list(colors)
            else:
                colors[idx_25] = old_c
            T *= 0.999
        
        if best_bad == 0:
            colors = best_colors
        else:
            n = 11
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    d = (j-i) % n
                    if d > n//2:
                        d = n - d
                    colors.append(d-1)
    
    return {'n': n, 'colors': colors}
