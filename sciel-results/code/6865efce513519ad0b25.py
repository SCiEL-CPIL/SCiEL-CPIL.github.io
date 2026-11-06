import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    # Lazebnik-Woldar construction: C4-free 5-coloring of K_26
    # This is a deterministic construction based on finite field F_9
    def lazebnik_woldar_n26():
        n = 26
        # Field elements: represent F_9 as {a + b*alpha} where alpha^2 = -1 = 2 (mod 3)
        # Actually, let's use the standard construction
        # F_9 = GF(3^2), elements: 0,1,2,3,4,5,6,7,8
        # We need to assign each pair (i,j) a color in {0,1,2,3,4}
        
        # The Lazebnik-Woldar construction uses the following:
        # Vertices are elements of F_9 x Z_3 (so 9*3=27, but we use 26)
        # Actually, the construction is more specific. Let me use a known valid coloring.
        
        # Alternative: use a known C4-free decomposition of K_26 into 5 graphs.
        # Since I can't easily reconstruct the exact algebraic construction in code quickly,
        # let me use a different approach: verify a known solution or use search with a good initial guess.
        
        # Actually, let me implement a more careful search approach that starts from
        # a structured initialization.
        
        # For now, return None to trigger the search path
        return None

    def count_violations(n, colors):
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost

    def solve_local_search(n, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
        
        total_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        color_matrix = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        def get_cost():
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0:
                        continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost
        
        current_cost = get_cost()
        if current_cost == 0:
            return colors
        
        search_start = time.time()
        stagnation_counter = 0
        best_cost_so_far = current_cost
        best_colors_snapshot = colors[:]
        temperature = 1.0
        cooling_rate = 0.995

        while time.time() - search_start < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = color_matrix[i][j]
            
            best_new_color = -1
            best_cost = current_cost
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                delta = 0
                
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_old][j] & (1 << w):
                        k = bin(neighbors[c_old][i] & neighbors[c_old][w]).count('1')
                        delta -= (k - 1)
                    if neighbors[c_old][i] & (1 << w):
                        k = bin(neighbors[c_old][j] & neighbors[c_old][w]).count('1')
                        delta -= (k - 1)
                
                for w in range(n):
                    if w == i or w == j:
                        continue
                    if neighbors[c_new][j] & (1 << w):
                        k = bin(neighbors[c_new][i] & neighbors[c_new][w]).count('1')
                        delta += k
                    if neighbors[c_new][i] & (1 << w):
                        k = bin(neighbors[c_new][j] & neighbors[c_new][w]).count('1')
                        delta += k
                        
                new_cost = current_cost + delta
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_new_color = c_new
            
            if best_new_color != -1:
                neighbors[c_old][i] &= ~(1 << j)
                neighbors[c_old][j] &= ~(1 << i)
                neighbors[best_new_color][i] |= (1 << j)
                neighbors[best_new_color][j] |= (1 << i)
                
                color_matrix[i][j] = best_new_color
                color_matrix[j][i] = best_new_color
                
                idx = i * (2 * n - i - 1) // 2 + j - i - 1
                colors[idx] = best_new_color
                
                current_cost = best_cost
                stagnation_counter = 0
                
                if current_cost < best_cost_so_far:
                    best_cost_so_far = current_cost
                    best_colors_snapshot = colors[:]
                
                if current_cost == 0:
                    return colors
            else:
                # Accept a random move with probability based on temperature (simulated annealing)
                if random.random() < temperature:
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    if ri == rj:
                        continue
                    if ri > rj:
                        ri, rj = rj, ri
                    rc_old = color_matrix[ri][rj]
                    rc_new = random.randint(0, 4)
                    if rc_new == rc_old:
                        continue
                    
                    neighbors[rc_old][ri] &= ~(1 << rj)
                    neighbors[rc_old][rj] &= ~(1 << ri)
                    neighbors[rc_new][ri] |= (1 << rj)
                    neighbors[rc_new][rj] |= (1 << ri)
                    
                    color_matrix[ri][rj] = rc_new
                    color_matrix[rj][ri] = rc_new
                    ridx = ri * (2 * n - ri - 1) // 2 + rj - ri - 1
                    colors[ridx] = rc_new
                    
                    current_cost = get_cost()
                    temperature *= cooling_rate
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    temperature *= cooling_rate
                    if stagnation_counter > 200:
                        # Restart with fresh random initialization
                        for _ in range(3):
                            ri = random.randint(0, n - 1)
                            rj = random.randint(0, n - 1)
                            if ri == rj:
                                continue
                            if ri > rj:
                                ri, rj = rj, ri
                            rc_old = color_matrix[ri][rj]
                            rc_new = random.randint(0, 4)
                            if rc_new == rc_old:
                                continue
                            
                            neighbors[rc_old][ri] &= ~(1 << rj)
                            neighbors[rc_old][rj] &= ~(1 << ri)
                            neighbors[rc_new][ri] |= (1 << rj)
                            neighbors[rc_new][rj] |= (1 << ri)
                            
                            color_matrix[ri][rj] = rc_new
                            color_matrix[rj][ri] = rc_new
                            ridx = ri * (2 * n - ri - 1) // 2 + rj - ri - 1
                            colors[ridx] = rc_new
                            current_cost = get_cost()
                            break
                        temperature = 0.5
                        stagnation_counter = 0

        return None

    # Try deterministic construction first
    n26_solution = lazebnik_woldar_n26()
    if n26_solution is not None and count_violations(26, n26_solution) == 0:
        return {'n': 26, 'colors': n26_solution}

    # Use local search with multiple restarts and simulated annealing
    best_n = 11
    best_colors = []
    
    # Initial seed solution: circulant coloring for small n
    def get_seed_solution(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append((d - 1) % 5)
        return colors

    best_colors = get_seed_solution(11)
    best_n = 11

    targets = [26, 25, 24, 23, 22]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Try multiple restarts for this n
        num_restarts = max(1, int(remaining / 5))
        found = False
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 1:
                break
                
            t_limit = min(4.0, deadline - time.time() - 0.5)
            if t_limit < 0.5:
                break
                
            sol = solve_local_search(n, t_limit, seed=42 + attempt * 7919)
            
            if sol is not None:
                if count_violations(n, sol) == 0:
                    best_n = n
                    best_colors = sol
                    found = True
                    break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
