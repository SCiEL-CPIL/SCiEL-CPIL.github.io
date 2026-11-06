import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 140.0  # Safety margin for 150s limit

    # Known seed solution for n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        # Build neighbor bitsets for each color
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        # Check for C4
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    def count_violations_and_edges(n, colors):
        """
        Counts violations and returns the list of edges involved in violations.
        A violation is a pair (u,v) with color c having >=2 common neighbors.
        We return the count and a set of edge indices that are "hot" (part of violations).
        """
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        hot_edges = set()
        
        # To identify hot edges, we need to know which edges contribute to the C4s.
        # If u and v have common neighbors x, y in color c, then the edges (u,x), (u,y), (v,x), (v,y) are involved.
        # However, changing any of these might fix it.
        # For simplicity in the polish phase, we can just pick random edges from the violation set.
        
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
                        # Mark edges incident to u or v in color c as hot?
                        # Or just store the pair (u,v,c)
                        # For the polish, we will pick edges incident to u or v.
                        # We'll just return the count for now, and handle selection in the solver.
        return cost

    def solve_sa(n, time_limit):
        """
        Runs Simulated Annealing with a Polish phase.
        """
        total_edges = n * (n - 1) // 2
        
        # Initial structured coloring: color(i,j) = (i + j) % 5
        # This often has fewer C4s than random.
        colors = [0] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors[idx] = (i + j) % 5
                idx += 1
        
        # Precompute edge endpoints
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1
        
        current_cost = count_violations_and_edges(n, colors)
        
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        start_time = time.time()
        
        # Polish threshold
        polish_threshold = 5
        
        while time.time() - start_time < time_limit:
            # If we are close, try to polish with backtracking
            if current_cost < polish_threshold and random.random() < 0.1:
                polished = try_polish(n, colors, edge_map, 1.0)
                if polished is not None:
                    return polished
                
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            colors[e_idx] = new_c
            new_cost = count_violations_and_edges(n, colors)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                colors[e_idx] = old_c
                
            T *= cooling_rate
            if T < 0.01:
                T = 1.0
                # Perturb best solution
                for _ in range(10):
                    e_idx = random.randint(0, total_edges - 1)
                    colors[e_idx] = random.randint(0, 4)
                current_cost = count_violations_and_edges(n, colors)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]

        return best_colors if best_cost == 0 else None

    def try_polish(n, colors, edge_map, time_limit):
        """
        A simple backtracking attempt to fix remaining violations.
        """
        current_cost = count_violations_and_edges(n, colors)
        if current_cost == 0:
            return colors
        
        # Find edges involved in violations
        # We'll just pick random edges and try to change them if they reduce cost.
        # This is a greedy local search.
        colors_copy = colors[:]
        start = time.time()
        
        while time.time() - start < time_limit:
            # Find a violation
            # Re-calculate neighbors
            neighbors = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors_copy[idx]
                    neighbors[c][i] |= (1 << j)
                    neighbors[c][j] |= (1 << i)
                    idx += 1
            
            violation_found = False
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0: continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            violation_found = True
                            # Try to change an edge incident to u or v
                            # Pick a random neighbor w of u in color c
                            # Or just pick a random edge in the graph
                            # Let's pick a random edge
                            e_idx = random.randint(0, len(colors_copy) - 1)
                            old_c = colors_copy[e_idx]
                            new_c = random.randint(0, 4)
                            if new_c == old_c: continue
                            
                            colors_copy[e_idx] = new_c
                            new_cost = count_violations_and_edges(n, colors_copy)
                            if new_cost < current_cost:
                                current_cost = new_cost
                                if current_cost == 0:
                                    return colors_copy
                            else:
                                colors_copy[e_idx] = old_c
                            break
                    if violation_found: break
                if violation_found: break
            
            if not violation_found:
                return colors_copy

        return None

    # Strategy:
    # Try n from 29 down to 12.
    
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        if n >= 27:
            t_alloc = 5.0
        elif n == 26:
            t_alloc = 60.0
        else:
            t_alloc = 10.0
            
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        res = solve_sa(n, t_alloc)
        
        if res:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}
                
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
