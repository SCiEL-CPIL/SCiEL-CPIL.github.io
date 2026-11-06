import numpy as np
import itertools
import time
import random
import sys

def get_k6_cycles():
    """
    Generate all unique sets of 6 edges that form a C6.
    We represent a C6 by the set of its 6 edges (normalized pairs).
    """
    cycles = set()
    # A C6 is a permutation of 6 distinct vertices v0..v5 with edges (v0,v1)..(v5,v0)
    # We fix v0=0 and permute the rest to generate unique cycle structures on {0..5}
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = (0,) + perm
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w: u, w = w, u
            edges.add((u, w))
        # The set of edges uniquely identifies the cycle graph C6
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles()

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. Keep a safety margin.
    deadline = start_time + 145.0 

    def build_cycle_data(n):
        """
        Precompute:
        1. c6_edges_arr: shape (num_cycles, 6), dtype int32. Indices of edges in the global edge list.
        2. edge_to_cycles: list of lists. For each edge index, list of (cycle_idx, pos_in_cycle).
        """
        num_edges = n * (n - 1) // 2
        
        # Map (i, j) with i < j to a linear index
        # Index = i*(n-1) - i*(i-1)//2 + (j-i-1)
        # This formula is for 0-based i, j where i < j.
        # Let's verify:
        # n=4. Edges: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5.
        # i=0: 0*(3) - 0 + (j-1) = j-1. (0,1)->0, (0,2)->1, (0,3)->2. Correct.
        # i=1: 1*(3) - 0 + (j-2) = 3 + j - 2 = j+1. (1,2)->3, (1,3)->4. Correct.
        # i=2: 2*(3) - 1 + (j-3) = 6 - 1 + j - 3 = j+2. (2,3)->5. Correct.
        
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        c6_edges_list = []
        # Iterate over all 6-vertex subsets
        for comb in itertools.combinations(range(n), 6):
            # For each C6 structure on 6 vertices
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)
        
        c6_edges_arr = np.array(c6_edges_list, dtype=np.int32)
        
        # Build edge -> cycles mapping
        edge_to_cycles = [[] for _ in range(num_edges)]
        for ci, edges in enumerate(c6_edges_list):
            for pos, e in enumerate(edges):
                edge_to_cycles[e].append((ci, pos))
                
        return c6_edges_arr, edge_to_cycles, get_idx, num_edges

    def solve_n(n, time_limit, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        num_edges = n * (n - 1) // 2
        
        # Build data
        c6_edges_arr, edge_to_cycles, get_idx, num_edges = build_cycle_data(n)
        num_cycles = len(c6_edges_arr)
        
        if num_cycles == 0:
            return [0] * num_edges

        # Initialize colors randomly
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        
        # Initialize cycle color counts
        # cycle_color_count[ci, c] = number of edges in cycle ci with color c
        cycle_color_count = np.zeros((num_cycles, 4), dtype=np.int32)
        for ci in range(num_cycles):
            # Vectorized update for initialization
            edges = c6_edges_arr[ci]
            for e in edges:
                cycle_color_count[ci, colors[e]] += 1
        
        # Initial score: count cycles where any color count is 6
        # is_mono[ci] = 1 if any count is 6
        # We can compute this efficiently:
        # max_count = np.max(cycle_color_count, axis=1)
        # is_mono = (max_count == 6)
        # score = np.sum(is_mono)
        
        def get_current_score():
            # A cycle is monochromatic if max count is 6
            return int(np.sum(np.max(cycle_color_count, axis=1) == 6))
        
        current_score = get_current_score()
        if current_score == 0:
            return colors.tolist()
        
        best_score = current_score
        best_colors = colors.copy()
        
        # Simulated Annealing
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        no_improve_count = 0
        max_no_improve = 2000
        
        # Precompute exp table? No, T changes.
        # Use random.random() for acceptance
        
        while time.time() < start_time + time_limit:
            # Select a random edge
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1
            
            delta = 0
            # Update affected cycles
            for (ci, pos) in edge_to_cycles[idx]:
                old_mono = 1 if np.max(cycle_color_count[ci]) == 6 else 0
                
                cycle_color_count[ci, old_color] -= 1
                cycle_color_count[ci, new_color] += 1
                
                new_mono = 1 if np.max(cycle_color_count[ci]) == 6 else 0
                delta += new_mono - old_mono
            
            # Acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[idx] = new_color
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    no_improve_count = 0
                    if best_score == 0:
                        return best_colors.tolist()
                else:
                    no_improve_count += 1
            else:
                # Revert changes
                for (ci, pos) in edge_to_cycles[idx]:
                    cycle_color_count[ci, old_color] += 1
                    cycle_color_count[ci, new_color] -= 1
            
            T *= cooling
            if no_improve_count > max_no_improve:
                # Restart with higher temperature
                T = max(T * 2, 1.0)
                no_improve_count = 0

        if best_score == 0:
            return best_colors.tolist()
        return None

    # Strategy:
    # 1. Try n=19 briefly (low probability, but high reward).
    # 2. Try n=18 aggressively (high probability of success if it exists).
    # 3. Try n=17 (guaranteed success, baseline).
    
    # Time allocation
    # n=19: 5s
    # n=18: 60s
    # n=17: 20s
    # Fallback: rest
    
    # Try n=19
    if time.time() < deadline - 80:
        res = solve_n(19, 5.0, seed=1001)
        if res is not None:
            return {'n': 19, 'colors': res}
            
    # Try n=18
    # This is the critical step for improvement.
    # We give it a generous budget.
    if time.time() < deadline - 20:
        # Use multiple restarts if possible?
        # A single long run is often better for SA.
        # Let's give it 60 seconds.
        t_start = time.time()
        time_limit = 60.0
        # Ensure we don't exceed deadline
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 10.0
            
        if time_limit > 2.0:
            res = solve_n(18, time_limit, seed=2002)
            if res is not None:
                return {'n': 18, 'colors': res}
            
    # Try n=17
    # This should be very fast.
    if time.time() < deadline - 5:
        t_start = time.time()
        time_limit = 10.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 2.0
        if time_limit > 1.0:
            res = solve_n(17, time_limit, seed=3003)
            if res is not None:
                return {'n': 17, 'colors': res}
                
    # Fallback: n=16
    if time.time() < deadline - 2:
        t_start = time.time()
        time_limit = 5.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 1.0
        if time_limit > 0.5:
            res = solve_n(16, time_limit, seed=4004)
            if res is not None:
                return {'n': 16, 'colors': res}

    # Hard fallback: n=9 (very easy)
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    result = construct()
    print(f"Found n={result['n']}")
