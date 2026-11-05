import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def solve_sa(n, max_time=10):
        m = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(m)]
        
        neighbors = [[[] for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].append(j)
                neighbors[c][j].append(i)
        
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    nu = neighbors[c][u]
                    nv = neighbors[c][v]
                    cnt = 0
                    for x in nu:
                        if x in nv:
                            cnt += 1
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
                        
        if bad_total == 0:
            return colors, True

        T = 1.0
        cooling = 0.999
        best_colors = list(colors)
        best_bad = bad_total
        current_bad = bad_total
        
        time_limit = start_time + max_time
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j, n)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if j in neighbors[old_c][k]:
                    old_cnt = common[old_c][i][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][i][k] = new_cnt
                    common[old_c][k][i] = new_cnt
                if i in neighbors[old_c][k]:
                    old_cnt = common[old_c][j][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][j][k] = new_cnt
                    common[old_c][k][j] = new_cnt
                if j in neighbors[new_c][k]:
                    old_cnt = common[new_c][i][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][i][k] = new_cnt
                    common[new_c][k][i] = new_cnt
                if i in neighbors[new_c][k]:
                    old_cnt = common[new_c][j][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][j][k] = new_cnt
                    common[new_c][k][j] = new_cnt
            
            new_bad = current_bad + delta
            
            if new_bad < current_bad or random.random() < math.exp(-delta / T):
                current_bad = new_bad
                colors[idx] = new_c
                try:
                    neighbors[old_c][i].remove(j)
                except ValueError:
                    pass
                try:
                    neighbors[old_c][j].remove(i)
                except ValueError:
                    pass
                neighbors[new_c][i].append(j)
                neighbors[new_c][j].append(i)
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = list(colors)
                    if best_bad == 0:
                        return best_colors, True
            else:
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if j in neighbors[old_c][k]:
                        common[old_c][i][k] += 1
                        common[old_c][k][i] += 1
                    if i in neighbors[old_c][k]:
                        common[old_c][j][k] += 1
                        common[old_c][k][j] += 1
                    if j in neighbors[new_c][k]:
                        common[new_c][i][k] -= 1
                        common[new_c][k][i] -= 1
                    if i in neighbors[new_c][k]:
                        common[new_c][j][k] -= 1
                        common[new_c][k][j] -= 1
            T *= cooling
        
        return best_colors, (best_bad == 0)

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    for n in range(26, 11, -1):
        if time.time() > deadline - 10:
            break
        found = False
        restarts = 5
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            random.seed(n * 1000 + r)
            colors, is_valid = solve_sa(n, max_time=20)
            if is_valid:
                best_n = n
                best_colors = colors
                found = True
                break
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
