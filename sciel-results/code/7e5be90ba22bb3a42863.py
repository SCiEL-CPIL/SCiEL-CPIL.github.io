import numpy as np
import itertools
import time
import random

def get_k6_cycles():
    """
    Generate all unique edge sets for C6 cycles in K6 (vertices 0..5).
    Returns a list of frozensets of edge tuples (u, v) with u < v.
    """
    cycles = set()
    # A C6 is a permutation of 1,2,3,4,5 appended to 0, forming 0-v1-v2-v3-v4-v5-0
    # We need to generate unique cycles.
    # It's easier to just iterate all permutations and normalize.
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = (0,) + perm
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w: u, w = w, u
            edges.add((u, w))
        # Normalize the set of edges to a frozenset of tuples
        # The set of edges uniquely identifies the cycle in K6
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def get_idx(i, j, n):
        # Index of edge (i, j) in lexicographic order for K_n
        # i < j
        # Count edges (0, 1..i-1) + (1, 2..i) + ... + (i-1, i..j-1) + (i, i+1..j-1)
        # Number of edges starting with k < i is k * (n - k - 1) / 2? No.
        # Standard formula: i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_cycle_data(n):
        num_edges = n * (n - 1) // 2
        c6_edges_list = []
        # Iterate over all combinations of 6 vertices
        for comb in itertools.combinations(range(n), 6):
            # For each combination, map the local K6 cycles to global edges
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv, n))
                c6_edges_list.append(global_edges)
        
        if not c6_edges_list:
            return np.array([], dtype=np.int32).reshape(0, 6), num_edges
        
        c6_edges_arr = np.array(c6_edges_list, dtype=np.int32)
        return c6_edges_arr, num_edges

    def solve_n(n, time_limit, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        num_edges = n * (n - 1) // 2
        c6_edges_arr, num_edges_calc = build_cycle_data(n)
        num_cycles = c6_edges_arr.shape[0]
        
        if num_cycles == 0:
            # Trivial case, any coloring works
            return np.zeros(num_edges, dtype=np.int8).tolist()

        # Initialize colors
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        
        def evaluate(c):
            """Vectorized evaluation of monochromatic C6 count."""
            if num_cycles == 0:
                return 0
            c6_c = c[c6_edges_arr]
            # A cycle is monochromatic if min == max
            min_c = c6_c.min(axis=1)
            max_c = c6_c.max(axis=1)
            is_mono = (min_c == max_c)
            return int(np.sum(is_mono))

        current_score = evaluate(colors)
        if current_score == 0:
            return colors.tolist()
        
        best_score = current_score
        best_colors = colors.copy()
        
        # Simulated Annealing
        T = 5.0
        cooling = 0.9998
        min_T = 0.01
        
        while time.time() < start_time + time_limit:
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1
            
            colors[idx] = new_color
            new_score = evaluate(colors)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    if best_score == 0:
                        return best_colors.tolist()
            else:
                # Revert
                colors[idx] = old_color
            
            T *= cooling
            if T < min_T:
                # Restart or cool down
                T = 5.0
                # Optionally randomize slightly or full restart
                # Full restart is often better for this problem
                colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
                current_score = evaluate(colors)
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    if best_score == 0:
                        return best_colors.tolist()
        
        if best_score == 0:
            return best_colors.tolist()
        return None

    # Strategy: Try n=17 with multiple restarts
    # We use the time limit to run solve_n for n=17
    # If it fails, try n=16, etc.
    
    # Allocate time for n=17
    time_for_17 = 130.0
    if time.time() < start_time + time_for_17:
        result = solve_n(17, time_for_17, seed=42)
        if result is not None:
            return {'n': 17, 'colors': result}
    
    # If n=17 failed, try n=16 with remaining time
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 5:
            result = solve_n(16, remaining, seed=123)
            if result is not None:
                return {'n': 16, 'colors': result}
    
    # Fallback n=15
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 2:
            result = solve_n(15, remaining, seed=456)
            if result is not None:
                return {'n': 15, 'colors': result}
    
    # Fallback n=14
    if time.time() < deadline - 1:
        remaining = deadline - 1 - time.time()
        if remaining > 1:
            result = solve_n(14, remaining, seed=789)
            if result is not None:
                return {'n': 14, 'colors': result}

    # Hard fallback: return a valid small coloring
    # n=9 is usually easy
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}
