import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 150.0
    
    MAX_N = 40
    colors = np.zeros((MAX_N, MAX_N, MAX_N), dtype=np.int8)
    
    def get_initial_colors(n, seed):
        random.seed(seed)
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    colors[a][b][c] = random.randint(0, 1)
                    
    def calculate_energy(n):
        energy = 0
        for a, b, c, d in combinations(range(n), 4):
            if colors[a][b][c] == 0 and colors[a][b][d] == 0 and \
               colors[a][c][d] == 0 and colors[b][c][d] == 0:
                energy += 1
        for a, b, c, d, e in combinations(range(n), 5):
            if colors[a][b][c] == 1 and colors[a][b][d] == 1 and \
               colors[a][b][e] == 1 and colors[a][c][d] == 1 and \
               colors[a][c][e] == 1 and colors[a][d][e] == 1 and \
               colors[b][c][d] == 1 and colors[b][c][e] == 1 and \
               colors[b][d][e] == 1 and colors[c][d][e] == 1:
                energy += 1
        return energy

    def get_col_fast(u, v, w):
        if u > v: u, v = v, u
        if v > w: v, w = w, v
        if u > v: u, v = v, u
        return colors[u][v][w]

    def flip_triple(n, a, b, c):
        old_color = colors[a][b][c]
        new_color = 1 - old_color
        colors[a][b][c] = new_color
        
        delta_E = 0
        
        # Red K4 update
        for x in range(n):
            if x == a or x == b or x == c:
                continue
            
            c_abx = get_col_fast(a, b, x)
            c_acx = get_col_fast(a, c, x)
            c_bc_x = get_col_fast(b, c, x)
            
            was_red = (old_color == 0 and c_abx == 0 and c_acx == 0 and c_bc_x == 0)
            is_red = (new_color == 0 and c_abx == 0 and c_acx == 0 and c_bc_x == 0)
            
            if was_red and not is_red:
                delta_E -= 1
            elif not was_red and is_red:
                delta_E += 1
        
        # Blue K5 update
        others = [x for x in range(n) if x != a and x != b and x != c]
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                x = others[i]
                y = others[j]
                
                c_abx = get_col_fast(a, b, x)
                c_aby = get_col_fast(a, b, y)
                c_acx = get_col_fast(a, c, x)
                c_acy = get_col_fast(a, c, y)
                c_axy = get_col_fast(a, x, y)
                c_bc_x = get_col_fast(b, c, x)
                c_bc_y = get_col_fast(b, c, y)
                c_bxy = get_col_fast(b, x, y)
                c_cxy = get_col_fast(c, x, y)
                
                was_blue = (old_color == 1 and c_abx == 1 and c_aby == 1 and c_acx == 1 and 
                            c_acy == 1 and c_axy == 1 and c_bc_x == 1 and c_bc_y == 1 and 
                            c_bxy == 1 and c_cxy == 1)
                is_blue = (new_color == 1 and c_abx == 1 and c_aby == 1 and c_acx == 1 and 
                           c_acy == 1 and c_axy == 1 and c_bc_x == 1 and c_bc_y == 1 and 
                           c_bxy == 1 and c_cxy == 1)
                
                if was_blue and not is_blue:
                    delta_E -= 1
                elif not was_blue and is_blue:
                    delta_E += 1
        
        return delta_E

    def solve_for_n(n, max_iters=100000, initial_temp=1.0, cooling=0.9995):
        get_initial_colors(n, seed=random.randint(0, 10000))
        current_energy = calculate_energy(n)
        best_energy = current_energy
        best_colors = colors.copy()
        
        temp = initial_temp
        triples = list(combinations(range(n), 3))
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                break
            
            idx = random.randint(0, len(triples) - 1)
            a, b, c = triples[idx]
            
            delta_E = flip_triple(n, a, b, c)
            new_energy = current_energy + delta_E
            
            if new_energy < current_energy or random.random() < math.exp(-delta_E / temp):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return best_colors
            else:
                flip_triple(n, a, b, c)
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        return best_colors if best_energy == 0 else None

    best_n = 34
    best_colors = None
    
    for n in range(34, 41):
        if time.time() - start_time > time_limit - 20:
            break
        
        found = None
        for restart in range(3):
            if time.time() - start_time > time_limit - 20:
                break
            iters = 200000 if n == 34 else 100000
            result = solve_for_n(n, max_iters=iters)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_n = n
            best_colors = found

    if best_colors is None:
        n = 9
        colors_list = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors_list}
    
    colors_list = []
    for a, b, c in combinations(range(best_n), 3):
        colors_list.append(int(best_colors[a][b][c]))
        
    return {'n': best_n, 'colors': colors_list}
