import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety

    # Fallback: known valid n=9 coloring
    n_fallback = 9
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]

    best_n = n_fallback
    best_colors = colors_fallback[:]

    # Try n from 18 down to 10
    # Allocate time: give more time to larger n
    time_budgets = {18: 50, 17: 30, 16: 15, 15: 8, 14: 5, 13: 3, 12: 2, 11: 1, 10: 1}

    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        alloc = min(time_budgets.get(n, 2), remaining)
        if alloc < 1:
            continue

        solution = solve_for_n(n, time.time() + alloc)
        if solution is not None:
            best_n = n
            best_colors = solution[:]
            # Since we're going from high to low, first success is best.
            return {'n': best_n, 'colors': best_colors}

    return {'n': best_n, 'colors': best_colors}

def solve_for_n(n, deadline):
    """Try to find a valid 4-coloring of K_n with no monochromatic C6."""
    start_time = time.time()
    
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute all 6-cycles and which edges they contain
    cycles_for_edge = [[] for _ in range(num_edges)]
    all_cycles = []
    
    # Generate all 6-cycles
    vertices = list(range(n))
    seen_cycle_sets = set()
    
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        # Generate all distinct 6-cycles on these 6 vertices
        # Fix v0, permute the other 5
        # To avoid duplicates: a cycle (v0, a, b, c, d, e) is same as (v0, e, d, c, b, a)
        # So we can require a < e (or some other canonical form)
        
        for perm in itertools.permutations([v1, v2, v3, v4, v5]):
            # Canonical form: first element of perm < last element
            if perm[0] > perm[-1]:
                continue
            
            cycle_verts = [v0] + list(perm)
            edge_set = set()
            valid = True
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_set.add(idx)
            
            if len(edge_set) != 6:
                continue  # Not a simple cycle (shouldn't happen with distinct vertices)
            
            key = frozenset(edge_set)
            if key not in seen_cycle_sets:
                seen_cycle_sets.add(key)
                cycle_list = list(edge_set)
                all_cycles.append(cycle_list)
                for idx in cycle_list:
                    cycles_for_edge[idx].append(len(all_cycles) - 1)
    
    total_cycles = len(all_cycles)
    
    # If too many cycles, the problem is hard; but let's proceed
    # For n=18, C(18,6)=18564, each gives ~30 unique cycles (after symmetry), so ~550k cycles
    
    # Local search with simulated annealing
    def run_local_search(init_colors, time_limit):
        colors = init_colors[:]
        current_cost = count_mono_cycles_fast(colors, all_cycles)
        
        if current_cost == 0:
            return colors[:]
        
        best_colors = colors[:]
        best_cost = current_cost
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() < time_limit - 0.5:
            e_idx = random.randint(0, num_edges - 1)
            c_old = colors[e_idx]
            
            best_delta = 0
            best_c_new = c_old
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                delta = compute_delta(e_idx, c_old, c_new, colors, all_cycles, cycles_for_edge)
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
            
            if best_delta < 0:
                colors[e_idx] = best_c_new
                current_cost += best_delta
                if current_cost == 0:
                    return colors[:]
            else:
                # Metropolis acceptance
                if best_delta > 0:
                    if random.random() < pow(2.718281828, -best_delta / T):
                        colors[e_idx] = best_c_new
                        current_cost += best_delta
                else:
                    colors[e_idx] = best_c_new
                    current_cost += best_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_colors if best_cost == 0 else None
    
    def count_mono_cycles_fast(col, cycles):
        count = 0
        for cycle in cycles:
            c0 = col[cycle[0]]
            mono = True
            for e in cycle[1:]:
                if col[e] != c0:
                    mono = False
                    break
            if mono:
                count += 1
        return count
    
    def compute_delta(e_idx, c_old, c_new, colors, cycles, cycles_for_edge):
        delta = 0
        for c_idx in cycles_for_edge[e_idx]:
            cycle = cycles[c_idx]
            # Check if all other 5 edges are c_old (was mono, now not)
            # Check if all other 5 edges are c_new (was not mono, now mono)
            is_mono_old = True
            is_mono_new = True
            for e in cycle:
                if e == e_idx:
                    continue
                c = colors[e]
                if c != c_old:
                    is_mono_old = False
                if c != c_new:
                    is_mono_new = False
                if not is_mono_old and not is_mono_new:
                    break
            if is_mono_old:
                delta -= 1
            if is_mono_new:
                delta += 1
        return delta
    
    # Try multiple random initializations
    num_restarts = 5
    restart_time = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 1:
            break
        
        # Random initialization
        init = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Also try structured initialization based on n=9 solution if n >= 9
        if n >= 9 and r == 0:
            # Extend the n=9 coloring pattern
            init = extend_n9_coloring(n)
        
        result = run_local_search(init, min(time.time() + restart_time, deadline - 0.5))
        if result is not None:
            return result
    
    # Try a greedy backtracking approach for smaller n
    if n <= 12 and time.time() < deadline - 5:
        result = backtrack_solve(n, deadline)
        if result is not None:
            return result
    
    return None

def extend_n9_coloring(n):
    """Extend the known n=9 circular coloring to larger n."""
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    for i in range(n):
        for j in range(i + 1, n):
            # Use circular distance modulo 9 for vertices < 9, random for others
            if i < 9 and j < 9:
                d = min(j - i, 9 - j + i)
                colors[edge_index[(i, j)]] = d - 1
            else:
                # Heuristic: use distance mod 4 or random
                d = (j - i) % 4
                colors[edge_index[(i, j)]] = d
    
    return colors

def backtrack_solve(n, deadline):
    """Backtracking search for small n."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Adjacency matrices for each color
    adj = [[[0] * n for _ in range(n)] for _ in range(4)]
    colors = [-1] * num_edges
    
    def has_path_length_5(u, v, color, visited):
        """Check if there's a simple path of length 5 from u to v in color graph."""
        if depth_check(u, v, color, 5, visited):
            return True
        return False
    
    def depth_check(cur, target, color, depth, visited):
        if depth == 0:
            return cur == target
        if depth == 1:
            return adj[color][cur][target] == 1 and target not in visited
        for nxt in range(n):
            if nxt in visited:
                continue
            if nxt == target and depth > 1:
                continue
            if adj[color][cur][nxt] == 1:
                visited.add(nxt)
                if depth_check(nxt, target, color, depth - 1, visited):
                    visited.remove(nxt)
                    return True
                visited.remove(nxt)
        return False
    
    def is_valid_add(u, v, color):
        """Check if adding edge (u,v) with 'color' creates a C6."""
        # Need a simple path of length 5 from u to v in the current color graph (without (u,v))
        visited = {u}
        return depth_check(u, v, color, 5, visited)
    
    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        # Symmetry breaking
        if idx == 0:
            cs = [0]
        elif idx == 1:
            cs = [0, 1]
        elif idx == 2:
            cs = [0, 1, 2]
        else:
            cs = [0, 1, 2, 3]
        
        for c in cs:
            if is_valid_add(u, v, c):
                colors[idx] = c
                adj[c][u][v] = 1
                adj[c][v][u] = 1
                
                if bt(idx + 1):
                    return True
                
                adj[c][u][v] = 0
                adj[c][v][u] = 0
                colors[idx] = -1
        
        return False
    
    if bt(0):
        return colors[:]
    return None

# Entry point
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback
    n_fallback = 9
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    
    best_n = n_fallback
    best_colors = colors_fallback[:]
    
    time_budgets = {18: 50, 17: 35, 16: 15, 15: 8, 14: 5, 13: 3, 12: 2, 11: 1, 10: 1}
    
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        alloc = min(time_budgets.get(n, 2), remaining)
        if alloc < 1:
            continue
        
        solution = solve_for_n(n, time.time() + alloc)
        if solution is not None:
            best_n = n
            best_colors = solution[:]
            return {'n': best_n, 'colors': best_colors}
    
    return {'n': best_n, 'colors': best_colors}
