import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s for safety
    
    def generate_c6s(n):
        """Generate all C6 subgraphs in K_n."""
        c6_list = []
        for comb in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = comb
            # The 5 distinct Hamiltonian cycles on 6 vertices
            patterns = [
                [(v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)],
                [(v0,v1), (v1,v2), (v2,v4), (v4,v3), (v3,v5), (v5,v0)],
                [(v0,v1), (v1,v3), (v3,v2), (v2,v4), (v4,v5), (v5,v0)],
                [(v0,v1), (v1,v3), (v3,v4), (v4,v2), (v2,v5), (v5,v0)],
                [(v0,v1), (v1,v4), (v4,v2), (v2,v3), (v3,v5), (v5,v0)]
            ]
            for pat in patterns:
                c6_list.append(pat)
        return c6_list
    
    def solve(n, time_limit):
        if n < 6:
            return [0] * (n*(n-1)//2)
        
        # Build edge index map
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx_map[(i, j)] = idx
                idx += 1
        total_edges = idx
        
        # Generate C6s
        c6_list = generate_c6s(n)
        num_c6s = len(c6_list)
        
        if num_c6s == 0:
            return [0] * total_edges
        
        # Pre-compute c6_edge_indices
        c6_edge_indices = []
        for c6 in c6_list:
            indices = []
            for (u, v) in c6:
                key = (u, v) if u < v else (v, u)
                indices.append(edge_idx_map[key])
            c6_edge_indices.append(indices)
        
        # Map each edge to the list of C6 indices it participates in
        edge_to_c6s = [[] for _ in range(total_edges)]
        for i, c6_idx_list in enumerate(c6_edge_indices):
            for e_idx in c6_idx_list:
                edge_to_c6s[e_idx].append(i)
        
        # Initial random coloring
        colors = [random.randint(0, 3) for _ in range(total_edges)]
        
        # Calculate initial energy (number of monochromatic C6s)
        def calc_energy(colors):
            energy = 0
            for e_indices in c6_edge_indices:
                c0 = colors[e_indices[0]]
                is_mono = True
                for e in e_indices[1:]:
                    if colors[e] != c0:
                        is_mono = False
                        break
                if is_mono:
                    energy += 1
            return energy
        
        energy = calc_energy(colors)
        
        if energy == 0:
            return colors
        
        # Simulated Annealing
        T = 5.0
        T_end = 0.05
        alpha = 0.9995  # Cooling rate
        
        # Try to solve with simulated annealing
        while time.time() < start_time + time_limit:
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            old_color = colors[e_idx]
            
            # Try all 3 other colors, pick the best
            best_delta = float('inf')
            best_new_color = -1
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                delta = 0
                for c6_idx in edge_to_c6s[e_idx]:
                    e_indices = c6_edge_indices[c6_idx]
                    
                    # Check if C6 becomes monochromatic new_color
                    is_mono_new = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != new_color:
                            is_mono_new = False
                            break
                    if is_mono_new:
                        delta += 1
                    
                    # Check if C6 stops being monochromatic old_color
                    is_mono_old = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    if is_mono_old:
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_new_color == -1:
                continue
            
            # Accept or reject based on simulated annealing
            if best_delta < 0:
                colors[e_idx] = best_new_color
                energy += best_delta
                if energy == 0:
                    return colors
            elif best_delta == 0:
                colors[e_idx] = best_new_color
            else:
                # Accept worse move with probability exp(-delta/T)
                import math
                if random.random() < math.exp(-best_delta / T):
                    colors[e_idx] = best_new_color
                    energy += best_delta
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
        
        return None
    
    # Try n=17 first
    sol = solve(17, 100.0)
    if sol is not None:
        return {'n': 17, 'colors': sol}
    
    # If 17 fails, try 16
    sol = solve(16, 30.0)
    if sol is not None:
        return {'n': 16, 'colors': sol}
    
    # If 16 fails, try 15
    sol = solve(15, 15.0)
    if sol is not None:
        return {'n': 15, 'colors': sol}
    
    # Fallback to a known valid small graph
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
