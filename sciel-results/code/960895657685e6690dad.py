import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_result = None
    
    # Fallback from seed
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    def solve_for_n(n):
        if time.time() > deadline - 10:
            return None, None
            
        # Precompute edge indices
        edge_idx = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                edge_idx[u, v] = idx
                edge_idx[v, u] = idx
                idx += 1
        num_edges = idx
        
        # Precompute cycle structures (60 unique cycles on 6 vertices)
        cycles_struct = []
        seen = set()
        for perm in itertools.permutations(range(1, 6)):
            seq = (0,) + perm
            rev = (0,) + tuple(reversed(perm))
            if seq < rev:
                cycles_struct.append(seq)
        
        # Map to edge indices for all 6-subsets
        cycle_edges = []
        for subset in itertools.combinations(range(n), 6):
            for struct in cycles_struct:
                e1 = (subset[struct[0]], subset[struct[1]])
                e2 = (subset[struct[1]], subset[struct[2]])
                e3 = (subset[struct[2]], subset[struct[3]])
                e4 = (subset[struct[3]], subset[struct[4]])
                e5 = (subset[struct[4]], subset[struct[5]])
                e6 = (subset[struct[5]], subset[struct[0]])
                
                cycle_edges.append((
                    edge_idx[e1[0], e1[1]],
                    edge_idx[e2[0], e2[1]],
                    edge_idx[e3[0], e3[1]],
                    edge_idx[e4[0], e4[1]],
                    edge_idx[e5[0], e5[1]],
                    edge_idx[e6[0], e6[1]]
                ))
        
        cycle_edges = np.array(cycle_edges, dtype=np.int32)
        
        # Cost function
        def get_cost(col_arr):
            c_edges = col_arr[cycle_edges]
            mins = np.min(c_edges, axis=1)
            maxs = np.max(c_edges, axis=1)
            return np.sum(mins != maxs)
        
        # Simulated Annealing with Restarts
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        max_steps_no_improve = 20000
        steps_no_improve = 0
        
        while time.time() < deadline - 5:
            colors = np.random.randint(0, 4, size=num_edges)
            current_cost = get_cost(colors)
            best_cost = current_cost
            best_colors = colors.copy()
            steps_no_improve = 0
            
            while time.time() < deadline - 5 and steps_no_improve < max_steps_no_improve:
                if current_cost == 0:
                    return n, best_colors.tolist()
                
                e_idx = np.random.randint(0, num_edges)
                old_color = colors[e_idx]
                new_color = np.random.randint(0, 4)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                new_cost = get_cost(colors)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        steps_no_improve = 0
                    else:
                        steps_no_improve += 1
                else:
                    colors[e_idx] = old_color
                    steps_no_improve += 1
                    
                T *= cooling
                if T < T_min:
                    T = T_min
                    
            if best_cost == 0:
                return n, best_colors.tolist()
                
        return None, None

    # Try n=18
    if time.time() < deadline - 20:
        res_n, res_colors = solve_for_n(18)
        if res_n is not None:
            return {'n': res_n, 'colors': res_colors}

    # Try n=17
    if time.time() < deadline - 10:
        res_n, res_colors = solve_for_n(17)
        if res_n is not None:
            return {'n': res_n, 'colors': res_colors}

    # Fallback
    return {'n': n_seed, 'colors': seed_colors}
